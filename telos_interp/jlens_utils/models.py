"""Which model a lens run is about — the fifth registry.

`METHODS` says how a token is selected, `SCORES` how it is scored, `STRATEGIES` where a
rollout cuts, `PROBE_TYPES` what a probe predicts. This one says *whose residual stream*
all of that is reading, and it exists because four constants in the gather were spelled
`gpt-oss-20b` at a time when there was only one model.

THE PROBLEM IT SOLVES IS A NAME COLLISION, NOT A MODEL COUNT. Two names identify a model
here and they are not the same name:

  * the **serving id**, `model_params.model_id` in every trajectory JSON, which records
    what generated the trajectory -- `openai/gpt-oss-20b`, `gsarti/qwen3.6-35b`;
  * the **HF repo id**, what `from_pretrained` needs -- `openai/gpt-oss-20b`,
    `Qwen/Qwen3.6-35B-A3B`.

For gpt-oss the two are the same string, which is why the gather could read the serving id
straight out of the JSON and hand it to `from_pretrained` and nobody ever needed a flag.
For Qwen they differ: `gsarti/qwen3.6-35b` is a Together AI endpoint alias and is not a HF
repo at all. `aliases` is what closes that gap, so a Qwen trajectory resolves with no flag
and `--model-id` is only needed for a model this file has never seen.

WHY THE WEIGHT KEYS ARE WRITTEN OUT AND NOT MATCHED BY SUFFIX. The obvious resolver --
"the key ending in `norm.weight` that is not inside a decoder layer" -- is ambiguous on
Qwen3.6, which has three: the real `model.language_model.norm.weight`, `mtp.norm.weight`
(the multi-token-prediction head) and `model.visual.merger.norm.weight` (the vision
tower). All three are the right rank and a plausible shape, so guessing wrong yields
silently wrong logits rather than an error. The keys are therefore declared per model and
`validate_against_index` only *checks* them, listing the near misses when one is absent.

Stdlib only, like the rest of this package: the scripts that decide what lands on disk
import it without the model stack.
"""

from dataclasses import dataclass, field

__all__ = [
    "MODELS",
    "DEFAULT_MODEL_ID",
    "ModelSpec",
    "get_model",
    "known_models",
    "resolve_model",
    "validate_against_index",
]

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class ModelSpec:
    """Everything about a model that the lens pipeline hard-coded before this file.

    Args:
        model_id: the HF repo id, what `from_pretrained` is given.
        lens_file: the fitted Jacobian's filename inside `--jlens_dir`.
        unembed_file: the lm_head + final-norm cache's filename, beside it.
        lm_head_key / norm_key: safetensors keys for the unembed. Declared, never guessed.
        config_path: where the text config sits inside `config.json`. `()` is the top
            level; Qwen3.6 is a VLM wrapper and puts `num_hidden_layers` and
            `rms_norm_eps` under `text_config`, so a bare lookup raises KeyError.
        target_layer: the layer the Jacobian was fitted *toward*, where the lens is the
            identity and no `J` is applied. A property of the fit, so it is pinned here
            and checked against the config rather than derived from it.
        aliases: serving ids that mean this model.
    """

    model_id: str
    lens_file: str
    unembed_file: str
    lm_head_key: str
    norm_key: str
    config_path: tuple[str, ...] = ()
    target_layer: int | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        """The repo id's last segment, which is what every filename is built from.

        >>> MODELS["Qwen/Qwen3.6-35B-A3B"].name
        'Qwen3.6-35B-A3B'
        """
        return self.model_id.rsplit("/", 1)[-1]

    def text_config(self, config):
        """The sub-config holding `num_hidden_layers` / `rms_norm_eps`.

        Takes either the parsed `config.json` (a dict, which is what the unembed cache
        reads) or a live `PretrainedConfig` (an object, which is what the gather holds
        after `from_pretrained`), and returns the same kind it was given.

        >>> MODELS["openai/gpt-oss-20b"].text_config({"rms_norm_eps": 1e-5})
        {'rms_norm_eps': 1e-05}
        >>> MODELS["Qwen/Qwen3.6-35B-A3B"].text_config({"text_config": {"a": 1}})
        {'a': 1}
        """
        for key in self.config_path:
            config = config[key] if isinstance(config, dict) else getattr(config, key)
        return config

    def num_hidden_layers(self, config) -> int:
        """The text stack's depth, from either flavour of config.

        >>> MODELS["Qwen/Qwen3.6-35B-A3B"].num_hidden_layers({"text_config": {"num_hidden_layers": 40}})
        40
        """
        text = self.text_config(config)
        return text["num_hidden_layers"] if isinstance(text, dict) else text.num_hidden_layers

    def resolve_target_layer(self, num_hidden_layers: int) -> int:
        """The identity layer, checked against the loaded model rather than trusted.

        A `target_layer` that is not the last block means the lens file and the model do
        not belong together -- the commonest way for that to happen is a `--jlens_dir`
        left pointing at the other model's folder, which otherwise produces a full run of
        plausible numbers.

        >>> MODELS["openai/gpt-oss-20b"].resolve_target_layer(24)
        23
        >>> MODELS["Qwen/Qwen3.6-35B-A3B"].resolve_target_layer(40)
        39
        """
        derived = num_hidden_layers - 1
        if self.target_layer is None:
            return derived
        if self.target_layer != derived:
            raise ValueError(
                f"{self.model_id} pins target_layer={self.target_layer} but the loaded model has "
                f"{num_hidden_layers} layers (last block {derived}). The lens and the model do not "
                "match -- check --jlens_dir and --model-id."
            )
        return self.target_layer


MODELS: dict[str, ModelSpec] = {
    # The filenames are spelled out rather than derived. Both fit scripts write
    # `{name}_gridenv_jacobian_lens.pt`, but the gpt-oss lens deployed at
    # /workspace/jlens/gridenv is named without the `_gridenv` -- it was renamed at some
    # point -- and every tree on disk was gathered against that file. Deriving the name
    # here would quietly stop finding it.
    "openai/gpt-oss-20b": ModelSpec(
        model_id="openai/gpt-oss-20b",
        lens_file="gpt-oss-20b_jacobian_lens.pt",
        unembed_file="gpt-oss-20b_unembed.pt",
        lm_head_key="lm_head.weight",
        norm_key="model.norm.weight",
        target_layer=23,
    ),
    "Qwen/Qwen3.6-35B-A3B": ModelSpec(
        model_id="Qwen/Qwen3.6-35B-A3B",
        lens_file="Qwen3.6-35B-A3B_gridenv_jacobian_lens.pt",
        unembed_file="Qwen3.6-35B-A3B_unembed.pt",
        lm_head_key="lm_head.weight",
        # NOT model.norm.weight: Qwen3_5MoeForConditionalGeneration wraps the text stack,
        # so the final norm is one level in. See this module's docstring for the two
        # decoys that a suffix match would hit.
        norm_key="model.language_model.norm.weight",
        config_path=("text_config",),
        target_layer=39,
        aliases=("gsarti/qwen3.6-35b",),
    ),
}


def known_models() -> list[str]:
    """Every accepted spelling, repo ids and serving aliases alike, sorted."""
    return sorted({*MODELS, *(alias for spec in MODELS.values() for alias in spec.aliases)})


def get_model(name: str) -> ModelSpec:
    """The spec for a repo id or a serving alias.

    >>> get_model("gsarti/qwen3.6-35b").model_id
    'Qwen/Qwen3.6-35B-A3B'
    >>> get_model("openai/gpt-oss-20b").target_layer
    23
    """
    if name in MODELS:
        return MODELS[name]
    for spec in MODELS.values():
        if name in spec.aliases:
            return spec
    raise KeyError(f"unknown model {name!r}; known: {', '.join(known_models())}")


def resolve_model(trajectory_model_id: str, override: str | None = None) -> ModelSpec:
    """The model to load weights from, given the trajectory's own id and an optional flag.

    The override wins outright, so a trajectory whose serving id this file has never seen
    can still be gathered by naming the repo. Without one the serving id is resolved
    through `aliases`, which is why the Qwen trees need no flag.

    >>> resolve_model("gsarti/qwen3.6-35b").model_id
    'Qwen/Qwen3.6-35B-A3B'
    >>> resolve_model("some/unknown-endpoint", "openai/gpt-oss-20b").name
    'gpt-oss-20b'
    """
    return get_model(override or trajectory_model_id)


def validate_against_index(spec: ModelSpec, weight_map: dict) -> None:
    """Check the declared unembed keys exist, and say what is near when one does not.

    Raises rather than falling back to a suffix match: on a model with a multi-token
    prediction head or a vision tower the near misses are the same rank and a plausible
    shape, so a wrong pick is silently wrong logits for the whole run.

    >>> spec = MODELS["openai/gpt-oss-20b"]
    >>> validate_against_index(spec, {"lm_head.weight": "s1", "model.norm.weight": "s1"})
    >>> validate_against_index(spec, {"lm_head.weight": "s1", "model.language_model.norm.weight": "s1"})
    Traceback (most recent call last):
        ...
    KeyError: "openai/gpt-oss-20b: 'model.norm.weight' is not in the checkpoint index; keys ending in 'norm.weight': model.language_model.norm.weight"
    """
    for key in (spec.lm_head_key, spec.norm_key):
        if key in weight_map:
            continue
        suffix = key.rsplit(".", 2)[-2] + "." + key.rsplit(".", 1)[-1]
        near = sorted(k for k in weight_map if k.endswith(suffix) and ".layers." not in k)
        raise KeyError(
            f"{spec.model_id}: {key!r} is not in the checkpoint index; "
            f"keys ending in {suffix!r}: {', '.join(near) or 'none'}"
        )
