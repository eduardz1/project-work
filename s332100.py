#!/usr/bin/env python3

import os
import sys

# Add src to path so we can import tgp
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from Problem import Problem
from tgp.solution.solution import solution as _solution


def solution(p: Problem):
    return _solution(p)  # ty:ignore[invalid-argument-type]


if __name__ == "__main__":
    from tgp.__main__ import main

    main()
