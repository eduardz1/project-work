from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from tgp.cli.utils import (
    parse_args,
    print_benchmark_table,
    run_single_problem,
    save_results_json,
)


def main() -> None:
    args = parse_args()

    configs = [
        (n, a, b, d)
        for n in args.cities
        for a in args.alphas
        for b in args.betas
        for d in args.densities
    ]

    results = []

    console = Console()
    console.print(
        f"Running benchmark on {len(configs)} problem configurations with preset '{args.preset}'."
    )
    console.print()

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("", total=len(configs))

        for num_cities, alpha, beta, density in configs:
            progress.update(
                task,
                description=f"[cyan]Cities={num_cities}, α={alpha:.2f}, β={beta:.2f}, ρ={density:.2f}",
            )

            result = run_single_problem(
                num_cities,
                alpha,
                beta,
                density,
                args.seed,
                args.include_path,
                args.preset,
            )
            results.append(result)

            progress.advance(task)

    console.print()

    print_benchmark_table(results)

    if args.output:
        console.print()
        save_results_json(results, args.output)


if __name__ == "__main__":
    main()
