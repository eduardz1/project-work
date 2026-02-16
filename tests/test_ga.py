from tests.validate import validate_solution
from tgp.problem.problem import Problem
from tgp.solution.ga.common import evaluate_solution
from tgp.solution.solution import solution


def test_ga_beta_gt_one():
    P = Problem(num_cities=10, alpha=1.0, beta=2.0, density=0.5, seed=1)

    sol = solution(P)
    total_cost = evaluate_solution(
        sol,
        P.dists,
        P.alpha,
        P.beta,
    )
    assert isinstance(total_cost, float)
    assert total_cost > 0.0

    valid, errors = validate_solution(sol, P.graph)
    assert valid, (
        f"Solution is invalid: {errors}, first 10 elements of path are {sol[:10]}"
    )


def test_ga_beta_eq_one():
    P = Problem(num_cities=10, alpha=1.0, beta=1.0, density=0.5, seed=1)

    sol = solution(P)
    total_cost = evaluate_solution(
        sol,
        P.dists,
        P.alpha,
        P.beta,
    )
    assert isinstance(total_cost, float)
    assert total_cost > 0.0

    valid, errors = validate_solution(sol, P.graph)
    assert valid, (
        f"Solution is invalid: {errors}, first 10 elements of path are {sol[:10]}"
    )


def test_ga_beta_lt_one():
    P = Problem(num_cities=10, alpha=1.0, beta=0.5, density=0.5, seed=1)

    sol = solution(P)
    total_cost = evaluate_solution(
        sol,
        P.dists,
        P.alpha,
        P.beta,
    )
    assert isinstance(total_cost, float)
    assert total_cost > 0.0

    valid, errors = validate_solution(sol, P.graph)
    assert valid, (
        f"Solution is invalid: {errors}, first 10 elements of path are {sol[:10]}"
    )
