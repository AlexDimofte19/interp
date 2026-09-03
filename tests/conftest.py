"""Shared stubs and fixtures for the jlens script tests.

The real scripts need gpt-oss-20b on a GPU, so everything heavy is stubbed: a 24-layer
module list whose "residual stream" is a closed-form function of (layer, token id,
position), a 64-token unembed, and a hand-built jacobian lens. That is enough to pin down
the parts that can silently go wrong -- which rows of a padded batch a token's activation
comes from, which `.pt` filename it lands in, and what order the CSV rows are written in.

The stub's activations depend only on (layer, token id, position), never on the other rows
of the batch, so a batched run and an unbatched run must agree *exactly*; any disagreement
is a bug in the offset arithmetic rather than float noise.

These live here rather than in one test module because
tests/test_delete_non_jlens_selected.py needs the same tree: its whole point is that
pruning a full gather lands on what a filtered gather would have written.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

jrt = importlib.import_module("scripts.jlens_reasoning_tokens")

# Layer 19 has no jlens matrix, so it gets no CSV rows and cannot be scored -- the filter's
# candidate pool is the *scorable* layers, not everything --layers asked for. Both the
# gather script and the pruner derive it the same way, which is what keeps them landing on
# the same files.
SCORABLE_LAYERS = [20, 21, 22, 23]


DIM = 8
VOCAB = 64
N_LAYERS = 24
ACTION_IDS = {"RIGHT": 3, "LEFT": 4, "UP": 5, "DOWN": 6}


def activation_value(layer: int, token_id: int, position: int) -> torch.Tensor:
    """What the stub's layer `layer` emits for `token_id` at `position`."""
    base = layer * 1000.0 + token_id * 10.0 + position * 0.5
    return base + torch.arange(DIM, dtype=torch.float32)


class _Block(torch.nn.Module):
    def __init__(self, idx: int) -> None:
        super().__init__()
        self.idx = idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        positions = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
        ids = input_ids.to(torch.float32).view(batch, seq, 1)
        dims = torch.arange(DIM, dtype=torch.float32).view(1, 1, DIM)
        return self.idx * 1000.0 + ids * 10.0 + positions * 0.5 + dims


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block(i) for i in range(N_LAYERS))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        # each block reads the ids directly, so an activation never depends on the
        # other sequences sharing the batch
        out = None
        for block in self.layers:
            out = block(input_ids)
        return out


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"num_hidden_layers": N_LAYERS})()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class _StubTokenizer:
    """Decodes id N to "<N>" and encodes that back, so the round trip is exact.

    `resolve_direction_ids` keeps a direction string only when it encodes to one token that
    decodes back to it -- the direction-mass table gathers logit *columns*, so a string
    without a single unambiguous id has no column to gather. Anything not of the form "<N>"
    encodes to two tokens here and is therefore dropped, which is the case worth having a
    stub for.
    """

    def decode(self, ids):
        return f"<{ids[0]}>"

    def encode(self, text, add_special_tokens=False):
        if text.startswith("<") and text.endswith(">") and text[1:-1].isdigit():
            return [int(text[1:-1])]
        return [0, 0]


def _token(token_id: int, groups=()):
    return {"token_id": token_id, "token": f"t{token_id}", "token_groups": list(groups)}


def _trajectory(step_output_lengths):
    """A trajectory whose steps have deliberately different reasoning lengths.

    Uneven lengths force real padding in any group of more than one step, which is the
    case the batched path has to get right.
    """
    steps = []
    for step_id, n_out in enumerate(step_output_lengths):
        output = [_token(20 + i, ["analysis"]) for i in range(n_out)]
        output.append(_token(40, ["final", "action"]))  # not a reasoning token
        steps.append(
            {
                "step_id": step_id,
                "agent_action": ["UP", "DOWN", "LEFT"][step_id % 3],
                # grid width varies with the step so output_start differs too
                "grid_state_tokens": [_token(50 + j) for j in range(2 + step_id)],
                "output_tokens": output,
            }
        )
    return {
        "model_params": {"model_id": "stub/model"},
        "grid_params": {},
        "prompt": {
            "prompt_prefix_tokens": [_token(1), _token(2)],
            "prompt_suffix_tokens": [_token(7)],
        },
        "steps": steps,
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Trajectory + jlens assets on disk, with the model and unembed stubbed out."""
    traj_dir = tmp_path / "trajectories" / "size5"
    traj_dir.mkdir(parents=True)
    stem = "stub_size5_comp0.8_1"
    traj_path = traj_dir / f"{stem}.json"
    traj_path.write_text(json.dumps(_trajectory([4, 2, 5])))

    jlens_dir = tmp_path / "jlens"
    jlens_dir.mkdir()
    torch.manual_seed(0)
    # layers 20-22 get a lens; 19 gets none (activations but no CSV rows) and 23 is
    # TARGET_LAYER, where the lens is the identity
    lens = {"J": {layer: torch.randn(DIM, DIM) for layer in (20, 21, 22)}}
    torch.save(lens, jlens_dir / "gpt-oss-20b_jacobian_lens.pt")

    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _StubModel()),
    )
    sampled = importlib.import_module("scripts.jlens_action_ranks_sampled")
    monkeypatch.setattr(sampled, "action_token_ids", lambda: (ACTION_IDS, _StubTokenizer()))
    # built once: every run in a test must see the same unembed, or two runs of the same
    # trajectory would differ for reasons that have nothing to do with the code under test
    assets = {
        "lm_head": torch.randn(VOCAB, DIM),
        "norm_weight": torch.randn(DIM),
        "rms_eps": 1e-5,
    }
    monkeypatch.setattr(sampled, "ensure_unembed_assets", lambda _dir: assets)
    return {"traj_path": traj_path, "jlens_dir": jlens_dir, "stem": stem, "tmp": tmp_path}


def _run(env, out_name, *extra):
    out = env["tmp"] / out_name
    argv = [
        "jlens_reasoning_tokens.py",
        "--trajectory-paths",
        str(env["traj_path"]),
        "--jlens_dir",
        str(env["jlens_dir"]),
        "--activations-dir",
        str(out),
        "--layers",
        "19:23",
        "--steps",
        "all",
        "--device",
        "cpu",
        "--torch-dtype",
        "float32",
        *extra,
    ]
    old = sys.argv
    sys.argv = argv
    try:
        jrt.main()
    finally:
        sys.argv = old
    return out / "size5" / env["stem"]


@pytest.fixture
def signal_json(env):
    """A 'direction' vocabulary drawn from the stub tokenizer's decoded strings.

    Which tokens score highest is arbitrary here (the lens is random), but it is
    deterministic, which is all the filter needs to be tested against.
    """
    path = env["tmp"] / "signal.json"
    path.write_text(
        json.dumps(
            {
                "UP": [f"<{i}>" for i in range(0, 8)],
                "DOWN": [f"<{i}>" for i in range(8, 16)],
                "LEFT": [f"<{i}>" for i in range(16, 24)],
                "RIGHT": [f"<{i}>" for i in range(24, 32)],
            }
        )
    )
    return path


def _select_args(signal_json, **overrides):
    opts = {"num-tokens": 2, "num-layers": 2, "always-layers": "", "random-tokens": 1, "seed": 42}
    opts.update(overrides)
    args = ["--signal-json", str(signal_json)]
    for key, value in opts.items():
        args += [f"--select-{key}", str(value)]
    return args
