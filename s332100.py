import sys
import os

# Add src to path so we can import tgp
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tgp.problem.problem import Problem as BaseProblem
from tgp.solution.solution import solution as solve_problem


class Problem(BaseProblem):
    def solution(self):
        sol = solve_problem(self)
        # Ensure it ends with (0, 0) as per requirements
        if sol and sol[-1] != (0, 0):
            sol.append((0, 0))
        return sol
