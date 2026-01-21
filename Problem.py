from tgp.problem.problem import Problem as _Problem
from tgp.solution.baseline import greedy


class Problem(_Problem):
    def baseline(self):
        return greedy(self)
