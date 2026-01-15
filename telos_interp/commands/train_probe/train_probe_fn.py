"""Train and apply probing classifiers on model activations."""

import json
from dataclasses import dataclass
from typing import Union

import nnsight
import pandas as pd

from telos_interp import activations, probing, probing_gpu

from .train_probe_utils import (
    get_default_probe_output_dir,
    get_jsonl_predictions_output_path,
    get_predictions_output_path,
    print_evaluation_results,
    save_evaluation_results,
    save_predictions,
    validate_layer,
    validate_paths,
)


@dataclass
class TrainMulticlassProbeConfig:
    """Train a multi-class probing classifier on grid cell activations.

    Args:
        activations_dir: Directory containing activation files (acts_wall.pt, acts_empty.pt, etc.)
        layer: Layer number for the probe
        output_dir: Directory to save the probe
        eval_split: Fraction of data to use for evaluation
        reg_coeff: Regularization coefficient
        normalize: Whether to normalize activations
        fit_intercept: Whether to fit an intercept term
        verbose: Verbosity level (0=silent, 1+=show progress)
        use_gpu: Use GPU-accelerated PyTorch implementation
        learning_rate: Learning rate for GPU training
        num_epochs: Number of epochs for GPU training
        balanced_weights: Use balanced class weights to handle class imbalance
        balance_classes: Balance classes by downsampling the majority classes
        use_mlp: Use MLP architecture instead of linear (GPU only)
        mlp_hidden_size: Hidden size for MLP architecture (GPU only)
        batch_size: Batch size for GPU training
        observability: Observability of the grid
    """

    activations_dir: str
    layer: int
    output_dir: str | None = None
    eval_split: float = 0.2
    reg_coeff: float = 1e3
    normalize: bool = False
    fit_intercept: bool = True
    verbose: int = 0
    use_gpu: bool = False
    learning_rate: float = 0.1
    num_epochs: int = 100
    balanced_weights: bool = False
    balance_classes: bool = False
    use_mlp: bool = False
    mlp_hidden_size: int = 1024
    batch_size: int = 16384
    observability: activations.Observability = activations.Observability.full

    def __call__(self):
        class_weight = "balanced" if self.balanced_weights else None
        validate_layer(self.layer)

        if self.use_gpu:
            saved_probe_path = probing_gpu.train_and_save_multiclass_probe_gpu(
                self.activations_dir,
                self.layer,
                self.output_dir,
                self.eval_split,
                self.reg_coeff,
                self.normalize,
                self.fit_intercept,
                self.verbose,
                device=None,
                learning_rate=self.learning_rate,
                num_epochs=self.num_epochs,
                class_weight=class_weight,
                balance_classes=self.balance_classes,
                use_mlp=self.use_mlp,
                mlp_hidden_size=self.mlp_hidden_size,
                batch_size=self.batch_size,
                observability=self.observability,
            )
        else:
            if self.use_mlp:
                print("Warning: --use-mlp is only supported with --gpu flag. Ignoring MLP option.")
            saved_probe_path = probing.train_and_save_multiclass_probe(
                self.activations_dir,
                self.layer,
                self.output_dir,
                self.eval_split,
                self.reg_coeff,
                self.normalize,
                self.fit_intercept,
                self.verbose,
                class_weight=class_weight,
            )
        print(f"Multi-class probe saved to {saved_probe_path}")


@dataclass
class ProbePredictConfig:
    """Predict using the multiclass probe on a csv dataset of trajectories.

    Args:
        model_name_or_path: Model name or path to load
        probe_path: Path to the trained probe
        csv_path: Path to the CSV dataset
        layer: Layer number used by the probe
        results_path: Path to save the resulting json file
        observability: Observability of the grid
        observation_type: Type of observation
    """

    model_name_or_path: str
    probe_path: str
    csv_path: str
    layer: int
    results_path: str | None = None
    observability: activations.Observability = activations.Observability.full
    observation_type: activations.ObservationType = activations.ObservationType.full_prompt

    def __call__(self):
        validate_paths(self.probe_path, self.csv_path)
        validate_layer(self.layer)

        results_path = self.results_path
        if results_path is None:
            results_path = get_predictions_output_path(
                self.csv_path,
                self.probe_path,
                self.layer,
                self.observability.value,
                self.observation_type.value,
            )

        activations_list = activations.get_activations_for_each_row_in_csv(
            self.model_name_or_path, self.csv_path, self.layer, self.observability, self.observation_type
        )

        predictions = probing_gpu.predict_from_activations_list(self.probe_path, activations_list)
        evaluation_results = probing_gpu.evaluate_predictions(self.csv_path, predictions)

        print_evaluation_results(evaluation_results)
        save_predictions(predictions, results_path)
        save_evaluation_results(evaluation_results, results_path)


@dataclass
class ProbeEvaluatePredictionsConfig:
    """Evaluate probe predictions saved in probe-predict output file.

    Args:
        predictions_path: Path to the predictions JSON file
        original_csv_path: Path to the original CSV file used for predictions
    """

    predictions_path: str
    original_csv_path: str

    def __call__(self):
        with open(self.predictions_path) as f:
            predictions = json.load(f)
        evaluation_results = probing_gpu.evaluate_predictions(self.original_csv_path, predictions)

        print_evaluation_results(evaluation_results)
        save_evaluation_results(evaluation_results, self.predictions_path)


@dataclass
class ProbeEvalOnJsonlConfig:
    """Evaluate a probe on a JSONL file.

    Args:
        model_name_or_path: Model name or path to load
        probe_path: Path to the trained probe
        layer: Layer number used by the probe
        jsonl_path: Path to the JSONL file
        grid_size: Size of the grid
        observability: Observability of the grid
        observation_type: Type of observation
        results_path: Path to save the resulting json file
    """

    model_name_or_path: str
    probe_path: str
    layer: int
    jsonl_path: str
    grid_size: int
    observability: activations.Observability = activations.Observability.full
    observation_type: activations.ObservationType = activations.ObservationType.full_prompt
    results_path: str | None = None

    def __call__(self):
        results_path = self.results_path
        if results_path is None:
            results_path = get_jsonl_predictions_output_path(
                self.jsonl_path,
                self.grid_size,
                self.layer,
                self.observability.value,
                self.observation_type.value,
            )

        activations_list = activations.get_activations_for_each_row_in_jsonl(
            self.model_name_or_path,
            self.jsonl_path,
            self.grid_size,
            self.layer,
            self.observability,
            self.observation_type,
        )

        predictions = probing_gpu.predict_from_activations_list(self.probe_path, activations_list)

        df = pd.read_json(self.jsonl_path, lines=True)
        grid_size_idx = df[df["size"] == self.grid_size].index.tolist()

        assert len(grid_size_idx) == len(activations_list), "Number of grids and activations must match"
        for predictions_dict, idx in zip(predictions, grid_size_idx, strict=False):
            df.at[idx, "metadata"]["predictions"] = predictions_dict

        df.to_json(results_path, lines=True, orient="records", indent=4)
        print(f"Predictions saved to {results_path}")


@dataclass
class TrainProbeConfig:
    """Train a probing classifier on a dataset of activations.

    Args:
        positive_acts: Path to positive activations file
        negative_acts: Path to negative activations file
        layer: Layer number for the probe
        output_dir: Directory to save the probe
        eval_split: Fraction of data to use for evaluation
        reg_coeff: Regularization coefficient
        normalize: Whether to normalize activations
    """

    positive_acts: str
    negative_acts: str
    layer: int
    output_dir: str | None = None
    eval_split: float = 0.2
    reg_coeff: float = 1e3
    normalize: bool = True

    def __call__(self):
        output_dir = self.output_dir
        if output_dir is None:
            output_dir = get_default_probe_output_dir(self.positive_acts)

        saved_probe_path = probing.train_and_save_probe(
            self.positive_acts,
            self.negative_acts,
            self.layer,
            output_dir,
            self.eval_split,
            self.reg_coeff,
            self.normalize,
        )
        print(f"Probe saved to {saved_probe_path}")


@dataclass
class ApplyProbeConfig:
    """Apply a probe to a dataset of activations.

    Args:
        model_name_or_path: Model name or path to load
        probe_path: Path to the trained probe
        layer: Layer number used by the probe
        prompt: Prompt text
        response: Response text
        token_position: Position of the token to gather activations from
    """

    model_name_or_path: str
    probe_path: str
    layer: int
    prompt: str
    response: str
    token_position: activations.TokenPosition = activations.TokenPosition.response_avg

    def __call__(self):
        model = nnsight.LanguageModel(self.model_name_or_path, device_map="auto", dispatch=True)
        probe = probing.ProbingClassifier.load(self.probe_path)

        score = probing.apply_probe_on_prompt_response(
            model, probe, self.prompt, self.response, self.layer, self.token_position
        )
        print(f"Score: {score}")


TrainProbe = Union[
    TrainMulticlassProbeConfig,
    ProbePredictConfig,
    ProbeEvaluatePredictionsConfig,
    ProbeEvalOnJsonlConfig,
    TrainProbeConfig,
    ApplyProbeConfig,
]


def train_probe(config: TrainProbe):
    """Train and apply probing classifiers on model activations.

    Subcommands:
    - TrainMulticlassProbeConfig: Train a multi-class probe
    - ProbePredictConfig: Run predictions with a trained probe
    - ProbeEvaluatePredictionsConfig: Evaluate saved predictions
    - ProbeEvalOnJsonlConfig: Evaluate probe on JSONL data
    - TrainProbeConfig: Train a binary probe
    - ApplyProbeConfig: Apply probe to a single prompt/response
    """
    config()
