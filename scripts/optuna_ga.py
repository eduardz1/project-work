import argparse
import time

import numpy as np
import optuna
from rich.console import Console
from rich.table import Table

from tgp.problem.problem import Problem
from tgp.solution.baseline import greedy
from tgp.solution.ga.crossovers import iox, ox, pmx
from tgp.solution.ga.ga import ga
from tgp.solution.ga.mutations import inversion_mutation, swap_mutation

CROSSOVER_MAP = {
    "pmx": pmx,
    "ox": ox,
    "iox": iox,
}

MUTATION_MAP = {
    "swap": swap_mutation,
    "inversion": inversion_mutation,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune GA hyperparameters using Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ga-tuning",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_ga_tuning.db",
        help="Optuna storage URL (e.g., sqlite:///tuning.db)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for optimization",
    )
    return parser.parse_args()


def warmup_jit(problem_data):
    """
    Run a dummy trial to trigger Numba JIT compilation.

    Caching the Numba compilation is not an option as we get some weird behavior
    if we try this. I believe running a warmup trial is better either way
    because even with Numba caching the first run will have to JIT compile.
    """

    console = Console()
    console.print("[yellow]Running warmup trial for JIT compilation...[/yellow]")

    nodes, golds, dists, alpha, beta, _ = problem_data[0]

    _ = ga(
        nodes=nodes,
        dist_matrix=dists,
        golds=golds,
        alpha=alpha,
        beta=beta,
        population_size_percent=0.2,
        generations_percent=0.1,
        crossover=iox,
        mutation=swap_mutation,
    )

    console.print("[green]✓ JIT compilation complete[/green]\n")


def create_objective():
    # We can only define a small-ish set of problems as we have a lot of
    # hyperparameters to tune and running a large number of trials is more
    # beneficial compared to having a large set of problems.

    # This looks like a sufficiently challenging city set without being too large.
    city_sizes = [100]

    # We test both very low alpha which would allow us to take longer routes and
    # high alpha which would force us to take shorter routes.
    alphas = [0.05, 10.0]

    # Both low and high density graphs, should generalize well.
    densities = [0.2, 0.9]

    # We only optimize beta <= 1.0. For beta > 1.0 most of the gains come from
    # removing the bulk gold.
    betas = [0.05, 0.5, 1.0]

    problems = [
        Problem(num_cities=n, alpha=a, beta=b, density=d)
        for n in city_sizes
        for a in alphas
        for b in betas
        for d in densities
    ]

    problem_data = []
    for p in problems:
        nodes_array = np.array(list(p.graph.nodes), dtype=np.int32)
        golds_array = np.array(
            [p.graph.nodes[node]["gold"] for node in p.graph.nodes], dtype=np.float32
        )
        baseline_cost = greedy(p)
        problem_data.append(
            (nodes_array, golds_array, p.dists, p.alpha, p.beta, baseline_cost)
        )

    warmup_jit(problem_data)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = {
            "population_size_percent": trial.suggest_float(
                "population_size_percent", 0.2, 2.0
            ),
            "generations_percent": trial.suggest_float("generations_percent", 0.2, 2.0),
            "elitism_rate": trial.suggest_float("elitism_rate", 0.05, 0.4),
            "mutation_rate": trial.suggest_float("mutation_rate", 0.01, 0.3, log=True),
            "lns_rate": trial.suggest_float("lns_rate", 0.0, 0.5),
            "lns_num_to_remove_percent": trial.suggest_float(
                "lns_num_to_remove_percent", 0.02, 0.25
            ),
            "tournament_size_percent": trial.suggest_float(
                "tournament_size_percent", 0.01, 0.15
            ),
        }

        crossover_name = trial.suggest_categorical(
            "crossover", list(CROSSOVER_MAP.keys())
        )
        mutation_name = trial.suggest_categorical("mutation", list(MUTATION_MAP.keys()))

        crossover_fn = CROSSOVER_MAP[crossover_name]
        mutation_fn = MUTATION_MAP[mutation_name]

        all_improvements = []

        start_time = time.perf_counter()
        for nodes, golds, dists, alpha, beta, baseline_cost in problem_data:
            best_individual = ga(
                nodes=nodes,
                dist_matrix=dists,
                golds=golds,
                alpha=alpha,
                beta=beta,
                crossover=crossover_fn,
                mutation=mutation_fn,
                **params,
            )
            improvement = (baseline_cost - best_individual.fitness) / baseline_cost
            all_improvements.append(improvement)
        end_time = time.perf_counter()

        return -float(np.mean(all_improvements)), end_time - start_time

    return objective


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        f"[bold cyan]Starting Optuna study:[/bold cyan] {args.study_name}",
    )
    console.print(f"  Trials: {args.n_trials}")
    console.print(f"  Storage: {args.storage}")
    console.print(f"  Parallel jobs: {args.n_jobs}")
    console.print()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )
    study.set_metric_names(["-avg_improvement", "time_seconds"])

    objective = create_objective()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    console.print()
    console.print("[bold green]Optimization complete![/bold green]")
    console.print()

    best_trials = study.best_trials

    best_trial = min(best_trials, key=lambda t: t.values[0])

    table = Table(title="Best Hyperparameters (by improvement)", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    for param, value in best_trial.params.items():
        if isinstance(value, str):
            table.add_row(param, value)
        else:
            table.add_row(param, f"{value:.6f}")

    console.print(table)

    console.print()
    console.print(
        f"[bold]Best avg improvement:[/bold] {-best_trial.values[0] * 100:.2f}%"
    )
    console.print(f"[bold]Time for best trial:[/bold] {best_trial.values[1]:.2f}s")

    console.print()
    console.print(f"[bold]Pareto front size:[/bold] {len(best_trials)} trials")

    console.print()
    console.print("[bold]Copy-paste for ga() defaults:[/bold]")
    console.print(best_trial.params)


if __name__ == "__main__":
    main()
