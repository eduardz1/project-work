from tgp.problem.problem import Problem
from tgp.solution.baseline import greedy


def test_baseline_beta_gt_one():
    P = Problem(num_cities=10, alpha=1.0, beta=2.0, density=0.5, seed=1)

    base_cost = greedy(P)
    assert base_cost > 0.0


def test_baseline_beta_eq_one():
    P = Problem(num_cities=10, alpha=1.0, beta=1.0, density=0.5, seed=1)

    base_cost = greedy(P)
    assert base_cost > 0.0


def test_baseline_beta_lt_one():
    P = Problem(num_cities=10, alpha=1.0, beta=0.5, density=0.5, seed=1)

    base_cost = greedy(P)
    assert base_cost > 0.0
