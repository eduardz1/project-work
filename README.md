# Traveling Goblin Problem

To run the project, use `uv`:

```bash
uv run tgp
```

Due to Numba's JIT compilation, the first run will be slower, and compilation results are not cached. For better performance, run multiple simulations at once rather than running the program multiple times. To see all available options, run:

```bash
uv run tgp --help
```

For example, the following command runs simulations for all combinations of 100 cities with alphas 0.2 and 0.5, and betas 0.5 and 2.0:

```bash
uv run tgp --cities 100 --alpha 0.2 0.5 --beta 0.5 2.0
```

An `s332100.py` script and `base_requirements.txt` file are also provided to run the project without Python modules, as required by the assignment.

## Extra Libraries

Optional libraries are available for enhanced performance. Enable CUDA support for NVIDIA GPUs or Intel-specific optimizations for Intel CPUs using the `--extra` flag with `uv sync`. For example, to enable both CUDA and Intel support, run:

```bash
uv sync --extra gpu --extra intel
```

## Testing

To run the tests, use:

```bash
uv run pytest
```

## Development Environment

To set up the development environment, use:

```bash
uv sync --group dev  # add '--extra gpu' or '--extra intel' if needed
```

### Linting

Linting is done with `ruff` and `ty`. To lint the code, use:

```bash
uv run ruff check --fix
uv run ty check
```

### Hyperparameter Optimization

The `scripts/optuna_ga.py` script optimizes the Genetic Algorithm's hyperparameters using Optuna. It runs multiple trials and finds the Pareto front of solutions balancing solution quality and runtime. To run the optimization, use:

```bash
uv run scripts/optuna_ga.py --n-trials 100
```

You can also visualize the optimization results with the Optuna dashboard:

```bash
optuna-dashboard sqlite:///optuna_ga_tuning.db
```

This will start a web interface where you can explore the trial history, parameter importance, and optimization progress. Use `uv run scripts/optuna_ga.py --help` for additional options like changing the number of parallel jobs or study name.
