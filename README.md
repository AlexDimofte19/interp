# A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents

*Raghu Arghal, Fade Chen, Niall Dalton, Evgenii Kortukov, Calum McNamara, Angelos Nalmpantis, Moksh Nirvaan, Gabriele Sarti, Mario Giulianelli*

> **Abstract:** Understanding whether and how language model agents pursue goals is essential for ensuring the safety of AI systems deployed to act autonomously in the world. In this work, we study goal-directedness in a language model agent, GPT-OSS-20B, as it navigates procedurally generated 2D grid environments. We operationalize goal-directedness behaviourally--through the optimality of an agent's actions and through its robustness to environment perturbations--and representationally--by probing the agent's internal activations for evidence of structured spatial knowledge. Our behavioural evaluation reveals that GPT-OSS-20B generally acts as a goal-directed agent, navigating towards the goal across a range of grid sizes with above-chance optimality. Representationally, linear and MLP probes trained on the agent's residual stream activations at intermediate layers uncover internal representations that partially encode the spatial layout of the environment, including the positions of walls, the goal, and the agent itself. Taken together, our results indicate that GPT-OSS-20B can act as a goal-directed agent through reliance on internal representations that partially but non-trivially encode the spatial features of its environment.

Paper: [arxiv.org/abs/2602.08964](https://arxiv.org/abs/2602.08964)

## Citation

```bibtex
@article{arghal-etal-2026-behavioural,
    title={A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents},
    author={Raghu Arghal and Fade Chen and Niall Dalton and Evgenii Kortukov and Calum McNamara and Angelos Nalmpantis and Moksh Nirvaan and Gabriele Sarti and Mario Giulianelli},
    year={2026},
    journal={arXiv preprint arXiv:2602.08964},
    url={https://arxiv.org/abs/2602.08964}
}
```

## Installation

```bash
# Clone the repository
git clone https://github.com/SPAR-Telos/interp
cd interp

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .

# For vLLM-based activation extraction (requires GPU)
pip install -e ".[vllm]"
```

## Reproduction

The analysis pipeline has four stages. Each stage uses a CLI command provided by the `interp-cli` tool. See [`telos_interp/commands/README.md`](telos_interp/commands/README.md) for full documentation of all commands and options.

### 1. Gather activations

Extract model activations from trajectory JSON files:

```bash
interp-cli gather_activations \
    --trajectory-paths "data/trajectories/size5/*.json" \
    --output-dir data/activations/size5 \
    --layers all \
    --steps 0 \
    --output-indices -1
```

### 2. Prepare activations for probing

Format extracted activations into datasets suitable for probe training:

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir data/activations/size5 \
    --trajectories-dir data/trajectories/size5 \
    --probe-type grid_tile \
    --output-indices -1 \
    --balance-classes-per-trajectory
```

### 3. Train probes

Train cell identity classifiers or distance regression probes:

```bash
# Cell identity probe
interp-cli train_cognitive_map_probe \
    --train-data-path data/activations/size5/cognitive_map_activations_*.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --num-epochs 100

# Distance regression probe
interp-cli train_distance_probe \
    --train-data-path data/activations/size7/distance_activations_*.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --num-epochs 100
```

Example configuration files are provided in `configs/`.

### 4. Evaluate and apply probes

Evaluate probes on held-out data and apply them to generate trajectory-level predictions:

```bash
# Evaluate cell identity probe
interp-cli eval_cognitive_map_probe \
    --trajectories-dir data/trajectories/size5_test \
    --activations-dir data/activations/size5_test \
    --probe-path path/to/cognitive_map_probe.pt \
    --output-indices -1

# Apply probe to trajectories
interp-cli apply_cognitive_map_probe \
    --activations-dir data/activations/size5 \
    --trajectories-dir data/trajectories/size5 \
    --probe-path path/to/cognitive_map_probe.pt \
    --output-dir data/trajectories_with_probes/size5 \
    --layers 20 \
    --steps all \
    --output-indices -1
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and testing instructions.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
