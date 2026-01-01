import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rich.console import Console
from rich.table import Table

from tgp.cli.check_range import CheckRange
from tgp.problem.problem import Problem
from tgp.solution.baseline import greedy
from tgp.solution.ga.common import evaluate_solution
from tgp.solution.ga.presets import PRESETS
from tgp.solution.solution import solution
from tgp.types import SolutionType


@dataclass
class BenchmarkResult:
    num_cities: int
    alpha: float
    beta: float
    density: float
    solution_cost: float
    baseline_cost: float
    improvement_percent: float
    path: SolutionType | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="uv run tgp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Solve the Traveling Goblin Problem. In arguments that accept multiple values, the program will run each combination of the provided values as separate problem instances.",
        # These are 3.14
        # color=True,
        # suggest_on_error=True,
    )
    parser.add_argument(
        "--cities",
        type=int,
        nargs="+",
        default=[100],
        help="Number of cities in the graph",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        action=CheckRange,
        min=0.0,
        metavar="F",
        nargs="+",
        default=[0.5],
        help="Alpha parameter of the cost function",
    )
    parser.add_argument(
        "--betas",
        type=float,
        action=CheckRange,
        min=0.0,
        metavar="F",
        nargs="+",
        default=[2.0],
        help="Beta parameter of the cost function",
    )
    parser.add_argument(
        "--densities",
        type=float,
        action=CheckRange,
        inf=0.0,
        max=1.0,
        metavar="F",
        nargs="+",
        default=[0.5],
        help="Graph density, which controls the sparsity of the graph edges",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON",
    )
    parser.add_argument(
        "--include-path",
        action="store_true",
        help="Include solution path in the output JSON",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "quality"],
        default="fast",
        help="Choice of preset for the Genetic Algorithm hyperparameters",
    )
    return parser.parse_args()


def run_single_problem(
    num_cities: int,
    alpha: float,
    beta: float,
    density: float,
    seed: int,
    include_path: bool,
    preset: Literal["fast", "balanced", "quality"] = "balanced",
) -> BenchmarkResult:
    problem = Problem(
        num_cities=num_cities,
        alpha=alpha,
        beta=beta,
        density=density,
        seed=seed,
    )

    sol = solution(problem, **PRESETS[preset])

    total_cost = evaluate_solution(sol, problem.dists, problem.alpha, problem.beta)
    greedy_cost = greedy(problem)
    improvement = ((greedy_cost - total_cost) / greedy_cost) * 100

    return BenchmarkResult(
        num_cities=num_cities,
        alpha=alpha,
        beta=beta,
        density=density,
        solution_cost=float(total_cost),
        baseline_cost=float(greedy_cost),
        improvement_percent=float(improvement),
        path=sol if include_path else None,
    )


def print_benchmark_table(results: list[BenchmarkResult]):
    console = Console()

    table = Table(title="Benchmark Results")
    table.add_column("Cities", justify="right", style="cyan")
    table.add_column("Alpha", justify="right", style="cyan")
    table.add_column("Beta", justify="right", style="cyan")
    table.add_column("Density", justify="right", style="cyan")
    table.add_column("Solution Cost", justify="right")
    table.add_column("Baseline Cost", justify="right")
    table.add_column("Improvement %", justify="right")

    for result in results:
        # Map improvement [-100, 100] to RGB gradient: red -> yellow -> green
        pct = result.improvement_percent
        normalized = (max(-100, min(100, pct)) + 100) / 200

        if normalized < 0.5:
            r, g, b = 255, int(normalized * 2 * 255), 0
        else:
            r, g, b = int((1 - (normalized - 0.5) * 2) * 255), 255, 0

        table.add_row(
            str(result.num_cities),
            f"{result.alpha:.2f}",
            f"{result.beta:.2f}",
            f"{result.density:.2f}",
            f"{result.solution_cost:.2f}",
            f"{result.baseline_cost:.2f}",
            f"[rgb({r},{g},{b})]{pct:.2f}%[/rgb({r},{g},{b})]",
        )

    console.print(table)

    if len(results) > 1:
        improvements = [r.improvement_percent for r in results]
        console.print()
        console.print("[bold]Summary Statistics:[/bold]")
        console.print(f"  Mean improvement: {np.mean(improvements):.2f}%")
        console.print(f"  Median improvement: {np.median(improvements):.2f}%")
        console.print(f"  Min improvement: {np.min(improvements):.2f}%")
        console.print(f"  Max improvement: {np.max(improvements):.2f}%")
        console.print(f"  Std deviation: {np.std(improvements):.2f}%")


def save_results_json(results: list[BenchmarkResult], output_path: str):
    output_file = Path(output_path)

    data = {
        "results": [asdict(r) for r in results],
        "summary": {
            "total_problems": len(results),
            "mean_improvement": float(
                np.mean([r.improvement_percent for r in results])
            ),
            "median_improvement": float(
                np.median([r.improvement_percent for r in results])
            ),
            "min_improvement": float(np.min([r.improvement_percent for r in results])),
            "max_improvement": float(np.max([r.improvement_percent for r in results])),
            "std_improvement": float(np.std([r.improvement_percent for r in results])),
        },
    }

    with output_file.open("w") as f:
        json.dump(data, f, indent=2)

    console = Console()
    console.print(f"[green]✓ Results saved to {output_path}[/green]")
