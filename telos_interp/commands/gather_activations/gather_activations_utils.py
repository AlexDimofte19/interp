"""Utility functions for gathering activations."""

import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

# Known token groups from the trace viewer format
KNOWN_TOKEN_GROUPS = {
    "prompt",
    "template",
    "placeholder",
    "grid_state",
    "grid_tile",
    "output",
    "analysis",
    "final",
    "action",
}


def is_token_group_specification(spec: str) -> bool:
    """Check if a specification string refers to a token group rather than numeric indices.

    A specification is considered a token group if:
    - It matches a known token group name (case-insensitive)
    - Or it starts with '@' prefix (for custom/unknown groups)


        spec: Index specification string to check

    Returns:
        True if the spec refers to a token group, False if it's numeric indices
    """
    spec_stripped = spec.strip().lower()

    # Check for @ prefix (explicit token group marker)
    if spec_stripped.startswith("@"):
        return True

    # Check against known token groups
    return spec_stripped in KNOWN_TOKEN_GROUPS


def get_indices_for_token_group(tokens: list[dict], group_name: str) -> list[int]:
    """Get token indices where the specified group appears in token_groups.

    Args:
        tokens: List of token dictionaries, each with 'id' and 'token_groups' keys
        group_name: Name of the token group to filter by (with or without '@' prefix)

    Returns:
        Sorted list of token indices (from 'id' field) where the group is present

    Raises:
        ValueError: If no tokens match the specified group
    """
    # Remove @ prefix if present
    group_name = group_name.lstrip("@").strip()

    indices = []
    for token in tokens:
        token_groups = token.get("token_groups", [])
        if group_name in token_groups:
            indices.append(token["id"])

    if not indices:
        raise ValueError(f"No tokens found with token_group '{group_name}'")

    return sorted(indices)


_EOS_PUNCTUATION = {".", "!", "?"}
_BPE_TRAILING = "Ċ\n "


def get_indices_for_eos_tokens(tokens: list[dict]) -> list[int]:
    """Get token indices where decoded text ends a sentence.

    Detects tokens whose text ends with '.', '!', or '?' after stripping
    trailing BPE whitespace/newline artifacts ('Ċ', newline, space).

    Args:
        tokens: List of token dicts with 'id' and 'token' keys

    Returns:
        Sorted list of token indices at sentence endings (may be empty)
    """
    indices = []
    for token in tokens:
        text = token.get("token", "").rstrip(_BPE_TRAILING)
        if text and text[-1] in _EOS_PUNCTUATION:
            indices.append(token["id"])
    return sorted(indices)


def parse_list_of_integers(value: str) -> list[int]:
    """Parse a comma-separated string of integers into a list.

    Args:
        value: Comma-separated string of integers (e.g., "1,2,3")

    Returns:
        List of integers
    """
    return [int(x) for x in value.split(",")]


def get_default_hf_directory(jsonl_path: str) -> str:
    """Get default HuggingFace directory name from JSONL path.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        Basename of the JSONL file to use as directory name
    """
    return os.path.basename(jsonl_path)


def parse_index_specification(spec: str, max_index: int, clamp: bool = False) -> list[int]:
    """Parse index specification string into list of indices with negative indexing support.

    Supports:
        - "all" -> all indices from 0 to max_index-1
        - "0,1,5" -> specific indices
        - "0:10" -> range (inclusive)
        - "-1" -> last index (max_index-1)
        - "-3:-1" -> last 3 indices
        - "0,5:10,-1" -> mixed format

    Args:
        spec: Index specification string
        max_index: Maximum index (exclusive) for "all" and for resolving negative indices
        clamp: If True, clamp out-of-bounds indices to valid range instead of raising error.
            Useful for steps where "0:10" should mean "up to 10 or all available".

    Returns:
        Sorted list of unique indices (all non-negative)

    Raises:
        ValueError: If indices are out of range (when clamp=False) or spec is malformed
    """
    if spec.strip().lower() == "all":
        return list(range(max_index))

    indices = set()

    # Split by comma
    parts = spec.split(",")

    for part_raw in parts:
        part = part_raw.strip()
        if not part:
            continue

        # Check if this is a range (contains : separator)
        # A range looks like: "0:10", "-3:-1", "5:-1", "-5:10"
        if ":" in part:
            range_parts = part.split(":")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range specification: '{part}'")

            start_str, end_str = range_parts
            start_idx = int(start_str.strip())
            end_idx = int(end_str.strip())

            # Resolve negative indices
            if start_idx < 0:
                start_idx = max_index + start_idx
            if end_idx < 0:
                end_idx = max_index + end_idx

            if clamp:
                # Clamp to valid range
                start_idx = max(0, min(start_idx, max_index - 1))
                end_idx = max(0, min(end_idx, max_index - 1))
            else:
                # Validate
                if start_idx < 0 or start_idx >= max_index:
                    raise ValueError(f"Start index {start_str} out of bounds [0, {max_index})")
                if end_idx < 0 or end_idx >= max_index:
                    raise ValueError(f"End index {end_str} out of bounds [0, {max_index})")

            # Add range (inclusive)
            if start_idx <= end_idx:
                indices.update(range(start_idx, end_idx + 1))
            else:
                # Handle reversed ranges (e.g., "10:5" means 10,9,8,7,6,5)
                indices.update(range(end_idx, start_idx + 1))
        else:
            # Single index
            try:
                idx = int(part)
            except ValueError as e:
                raise ValueError(f"Invalid index specification: '{part}'") from e

            # Resolve negative index
            if idx < 0:
                idx = max_index + idx

            if clamp:
                # Clamp to valid range (skip if still out of bounds after clamping)
                if idx < 0 or idx >= max_index:
                    continue
            # Validate
            elif idx < 0 or idx >= max_index:
                raise ValueError(f"Index {part} out of bounds [0, {max_index})")

            indices.add(idx)

    return sorted(indices)


def sanitize_model_id(model_id: str) -> str:
    """Convert model_id to filesystem-safe format.

    Replaces '/' with '__' for filesystem compatibility.

    Args:
        model_id: Model identifier (e.g., "openai/gpt-oss-20b")

    Returns:
        Sanitized model_id (e.g., "openai__gpt-oss-20b")
    """
    return model_id.replace("/", "__")


def _get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the decoder block list across common HF causal-LM layouts.

    Replaces nnterp's name standardization: for gpt-oss / Llama-family models the
    blocks live at ``model.model.layers``. Other layouts are tried as fallbacks.

    Args:
        model: A HuggingFace causal-LM model

    Returns:
        The ModuleList of decoder blocks

    Raises:
        AttributeError: If no known decoder-layer path is found
    """
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Could not locate decoder layers on model")


def extract_activations_single_pass(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    token_indices: list[int],
    layer_indices: list[int],
) -> dict[int, dict[int, torch.Tensor]]:
    """Extract activations for specific tokens at specified layers in a single forward pass.

    Captures the raw residual-stream output of each requested decoder block via
    PyTorch forward hooks (equivalent to nnterp's ``layers_output[i]``).

    Args:
        model: HuggingFace causal-LM model
        input_ids: Tokenized input tensor of shape (1, seq_len), should be truncated
            to include only tokens up to the last position needed
        token_indices: List of absolute token indices to extract
        layer_indices: List of layer indices to extract

    Returns:
        Nested dict: layer_idx -> token_idx -> activation tensor
    """
    seq_len = input_ids.shape[1]
    valid_indices = [idx for idx in token_indices if idx < seq_len]
    if len(valid_indices) < len(token_indices):
        invalid = [idx for idx in token_indices if idx >= seq_len]
        print(f"Warning: Skipping {len(invalid)} token indices beyond sequence length {seq_len}")

    # Move input_ids to the model's device (important for device_map="auto" setups)
    try:
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
    except StopIteration:
        pass  # No parameters found, keep input_ids on CPU

    decoder_layers = _get_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # decoder blocks may return a tuple (hidden_states, ...) or a bare tensor
            captured[idx] = output[0] if isinstance(output, tuple) else output

        return hook

    for layer_idx in layer_indices:
        handles.append(decoder_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    results: dict[int, dict[int, torch.Tensor]] = {layer_idx: {} for layer_idx in layer_indices}
    for layer_idx in layer_indices:
        layer_output = captured[layer_idx]
        for token_idx in valid_indices:
            results[layer_idx][token_idx] = layer_output[0, token_idx, :].detach().cpu().clone()

    return results


def extract_activations_batched(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_indices: list[list[int]],
    layer_indices: list[int],
    keep_on_device: bool = False,
) -> dict[int, torch.Tensor]:
    """Extract activations for several sequences at once, stacked one tensor per layer.

    The batched counterpart of :func:`extract_activations_single_pass`. Instead of a
    ``{token_idx: (D,)}`` dict per layer it returns a single ``(N, D)`` tensor per layer,
    gathered with one ``index_select`` over the flattened ``(B*S, D)`` layer output. That
    replaces N separate device-to-host copies with one, and lets a caller that is about to
    do more GPU work keep the residual streams where they already are.

    Rows follow ``token_indices`` in order: sequence 0's positions, then sequence 1's, and
    so on, so a caller can slice the result back apart with a running offset.

    Sequences shorter than ``input_ids.shape[1]`` must be **right**-padded. Under causal
    attention a real token at position i only attends to positions <= i, all of which are
    real, so right padding cannot perturb the activations that are read back.

    Args:
        model: HuggingFace causal-LM model
        input_ids: (B, S) token ids, right-padded
        attention_mask: (B, S) mask, or None when nothing is padded
        token_indices: per sequence, the absolute positions to extract
        layer_indices: List of layer indices to extract
        keep_on_device: Leave the result on the model's device instead of copying to CPU

    Returns:
        Dict: layer_idx -> (N, D) tensor, N = total number of requested positions
    """
    seq_len = input_ids.shape[1]
    flat_indices: list[int] = []
    for row, positions in enumerate(token_indices):
        valid = [idx for idx in positions if idx < seq_len]
        if len(valid) < len(positions):
            print(f"Warning: Skipping {len(positions) - len(valid)} token indices beyond sequence length {seq_len}")
        flat_indices.extend(row * seq_len + idx for idx in valid)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = input_ids.device
    input_ids = input_ids.to(model_device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

    decoder_layers = _get_decoder_layers(model)
    index = torch.tensor(flat_indices, dtype=torch.long, device=model_device)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # decoder blocks may return a tuple (hidden_states, ...) or a bare tensor
            out = output[0] if isinstance(output, tuple) else output
            # gather here rather than after the pass so only the requested rows are kept
            captured[idx] = out.reshape(-1, out.shape[-1]).index_select(0, index).detach()

        return hook

    for layer_idx in layer_indices:
        handles.append(decoder_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if keep_on_device:
        return {layer_idx: captured[layer_idx] for layer_idx in layer_indices}
    return {layer_idx: captured[layer_idx].to("cpu") for layer_idx in layer_indices}


def build_truncated_input(
    trajectory: dict,
    step: dict,
    max_absolute_position: int,
) -> torch.Tensor:
    """Build input_ids truncated to the maximum position needed.

    Args:
        trajectory: Parsed trajectory JSON
        step: Step dictionary from trajectory
        max_absolute_position: The last absolute token position needed (inclusive)

    Returns:
        input_ids tensor of shape (1, max_absolute_position + 1)
    """
    prefix_ids = [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
    grid_ids = [t["token_id"] for t in step["grid_state_tokens"]]
    suffix_ids = [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
    output_ids = [t["token_id"] for t in step["output_tokens"]]

    # Concatenate all token IDs
    all_token_ids = prefix_ids + grid_ids + suffix_ids + output_ids

    # Truncate to only include tokens up to max_absolute_position (inclusive)
    truncated_ids = all_token_ids[: max_absolute_position + 1]

    return torch.tensor([truncated_ids])


def save_activations_to_files(
    activations: dict[int, dict[int, torch.Tensor]],
    output_base: Path,
    step_idx: int | None,
    category: str,
) -> int:
    """Save activations to .pt files in the output directory structure.

    Output structure:
        {output_base}/layer_{N}/step_{M}/{category}/{token_id}.pt
        or for step_idx=None:
        {output_base}/layer_{N}/{category}/{token_id}.pt

    Args:
        activations: Nested dict: layer_idx -> token_idx -> activation tensor
        output_base: Base output directory (includes model_id)
        step_idx: Step index (None for categories without step context)
        category: Token category name

    Returns:
        Number of activations with NaN values detected
    """
    writer = ActivationWriter(max_workers=0)
    writer.submit(activations, output_base, step_idx, category)
    return writer.drain()


def activation_layer_dir(output_base: Path, layer_idx: int, step_idx: int | None, category: str) -> Path:
    """The directory a token's .pt file lives in, for one layer/step/category.

    Args:
        output_base: Base output directory (includes model_id)
        layer_idx: Layer index
        step_idx: Step index (None for categories without step context)
        category: Token category name

    Returns:
        {output_base}/layer_{N}/step_{M}/{category}, or {output_base}/layer_{N}/{category}
    """
    if step_idx is not None:
        return output_base / f"layer_{layer_idx}" / f"step_{step_idx}" / category
    return output_base / f"layer_{layer_idx}" / category


class ActivationWriter:
    """Thread-pooled writer for the per-token .pt tree.

    Writes exactly what :func:`save_activations_to_files` writes — same paths, same
    tensors, same NaN accounting — but hands the files to a pool of worker threads, so
    the (many, tiny) file creations overlap with the caller's GPU work instead of
    blocking it. ``torch.save`` releases the GIL while writing, which is what makes this
    worth doing: a trajectory with a long reasoning chain writes one file per
    (token, layer), which is tens of thousands of files per step.

    ``max_workers=0`` writes inline on the calling thread, which is how
    :func:`save_activations_to_files` keeps its original synchronous behaviour.

    Every queued write must be drained before the run is recorded as complete:
    :meth:`drain` waits for the queue and re-raises the first worker exception, so a
    failed write can never be mistaken for a finished trajectory.
    """

    def __init__(self, max_workers: int = 16, use_zipfile: bool = True, max_pending: int = 4096) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 0 else None
        self._use_zipfile = use_zipfile
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._nan_count = 0
        self._pending: deque = deque()
        self._made_dirs: set[Path] = set()

    def __enter__(self) -> "ActivationWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def submit(
        self,
        activations: dict[int, dict[int, torch.Tensor]],
        output_base: Path,
        step_idx: int | None,
        category: str,
    ) -> None:
        """Queue one batch of activations for writing. Same arguments as save_activations_to_files.

        Tensors must not alias a larger storage — ``torch.save`` serialises the whole
        underlying storage, not the view — so slice callers should clone their rows first.
        """
        for layer_idx, token_activations in activations.items():
            layer_dir = activation_layer_dir(output_base, layer_idx, step_idx, category)
            # mkdir on the submitting thread: cheap, and keeps the workers race-free
            if layer_dir not in self._made_dirs:
                layer_dir.mkdir(parents=True, exist_ok=True)
                self._made_dirs.add(layer_dir)
            for token_idx, activation in token_activations.items():
                output_path = layer_dir / f"{token_idx}.pt"
                if self._pool is None:
                    self._write(output_path, activation)
                else:
                    self._throttle()
                    self._pending.append(self._pool.submit(self._write, output_path, activation))

    def _throttle(self) -> None:
        """Keep the queue bounded; a queued write holds its tensor alive.

        Futures complete roughly in submission order, so blocking on the oldest is enough
        to stop a fast producer from buffering a whole trajectory's activations in RAM.
        """
        if self._max_pending <= 0 or len(self._pending) < self._max_pending:
            return
        while len(self._pending) > self._max_pending // 2:
            self._pending.popleft().result()

    def _write(self, output_path: Path, activation: torch.Tensor) -> None:
        if torch.isnan(activation).any():
            with self._lock:
                self._nan_count += 1
        # the legacy (non-zip) container skips the zip directory per file; for the ~6 KB
        # tensors written here that overhead is a real fraction of the write. Both forms
        # still load under torch.load(..., weights_only=True).
        torch.save(activation, output_path, _use_new_zipfile_serialization=self._use_zipfile)

    def drain(self) -> int:
        """Wait for every queued write, then return (and reset) the NaN count.

        Raises whatever a worker raised, so callers never treat a failed write as done.
        """
        while self._pending:
            self._pending.popleft().result()
        self._made_dirs.clear()
        with self._lock:
            nan_count, self._nan_count = self._nan_count, 0
        return nan_count

    def close(self) -> int:
        """Drain, then shut the pool down. Returns the NaN count from the final drain."""
        try:
            return self.drain()
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None


def resolve_token_indices(
    spec: str,
    tokens: list[dict],
    category_name: str,
) -> list[int]:
    """Resolve token indices from either numeric specification or token group name.

    Args:
        spec: Index specification - either numeric (e.g., "-5:-1", "0,5,10") or token group name
        tokens: List of token dictionaries with 'id' and 'token_groups' keys
        category_name: Name of the token category (for error messages)

    Returns:
        List of token indices to extract
    """
    if spec.strip().lower() == "eos":
        indices = get_indices_for_eos_tokens(tokens)
        print(f"      EOS mode: {len(indices)} sentence-ending tokens in {category_name}")
        return indices
    elif is_token_group_specification(spec):
        group_name = spec.lstrip("@").strip()
        indices = get_indices_for_token_group(tokens, group_name)
        print(f"      Token group '{group_name}' matched {len(indices)} tokens in {category_name}")
        return indices
    else:
        return parse_index_specification(spec, len(tokens))
