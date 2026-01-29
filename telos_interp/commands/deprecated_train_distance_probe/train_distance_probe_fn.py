"""Train and apply distance regression probes on model activations."""

from dataclasses import dataclass

from telos_interp import distance_probing

from .train_distance_probe_utils import validate_layer, validate_paths


@dataclass
class TrainDistanceProbeConfig:
    """Train a distance regression probe on grid activations.

    Args:
        activations_dir: Directory containing raw prompt activations (all_layer_acts.pt)
        layer: Layer number for the probe
        eval_split: Fraction of data to use for evaluation
        reg_coeff: Regularization coefficient (weight decay)
        normalize: Whether to normalize activations
        learning_rate: Learning rate for training
        num_epochs: Number of epochs for training
        use_mlp: Use MLP architecture instead of linear
        mlp_hidden_size: Hidden size for MLP architecture
        batch_size: Batch size for training
        verbose: Verbosity level (0=silent, 1+=show progress)
        label_column: CSV column name for regression target
    """

    activations_dir: str
    layer: int
    eval_split: float = 0.2
    reg_coeff: float = 1e-4
    normalize: bool = False
    learning_rate: float = 1e-3
    num_epochs: int = 100
    use_mlp: bool = False
    mlp_hidden_size: int = 256
    batch_size: int = 1024
    verbose: int = 0
    label_column: str = "optimal_trajectory_length"

    def __call__(self):
        validate_layer(self.layer)

        saved_distance_probe_path = distance_probing.train_and_save_distance_probe(
            activations_dir=self.activations_dir,
            layer=self.layer,
            eval_split=self.eval_split,
            reg_coeff=self.reg_coeff,
            normalize=self.normalize,
            use_mlp=self.use_mlp,
            mlp_hidden_size=self.mlp_hidden_size,
            batch_size=self.batch_size,
            verbose=self.verbose,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            label_column=self.label_column,
        )
        print(f"Distance probe saved to {saved_distance_probe_path}")


@dataclass
class DistanceProbePredictConfig:
    """Predict distances using a trained distance probe on a CSV dataset.

    Gather activations from model, run distance probe predictions, and evaluate.

    Args:
        model_name_or_path: Model to use for gathering activations
        probe_path: Path to the trained distance probe
        csv_path: Path to CSV file with grid data and labels
        layer: Layer number used for the probe
        output_path: Path to save predictions JSON
        label_column: CSV column name for ground truth
    """

    model_name_or_path: str
    probe_path: str
    csv_path: str
    layer: int
    output_path: str | None = None
    label_column: str = "optimal_trajectory_length"

    def __call__(self):
        validate_paths(self.probe_path, self.csv_path)
        validate_layer(self.layer)

        predictions_path, eval_path = distance_probing.predict_and_evaluate_from_csv(
            model_name_or_path=self.model_name_or_path,
            probe_path=self.probe_path,
            csv_path=self.csv_path,
            layer=self.layer,
            label_column=self.label_column,
            output_path=self.output_path,
        )

        print(f"Predictions saved to {predictions_path}")
        print(f"Evaluation saved to {eval_path}")


TrainDistanceProbe = TrainDistanceProbeConfig | DistanceProbePredictConfig


def train_distance_probe(config: TrainDistanceProbe):
    """Train and apply distance regression probes on model activations.

    Subcommands:
    - TrainDistanceProbeConfig: Train a distance regression probe
    - DistanceProbePredictConfig: Run predictions with a trained distance probe
    """
    config()
