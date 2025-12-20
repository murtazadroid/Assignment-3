# backtest/baselines.py

import numpy as np
from typing import Protocol


class Policy(Protocol):
    """
    Interface for policies used in backtesting:
    they just need a select_action(obs) method.
    """
    def select_action(self, obs: np.ndarray) -> int:
        ...


class NoTradePolicy:
    """
    Baseline 1: never trades.
    Always returns action 0 (hold).
    """
    def select_action(self, obs: np.ndarray) -> int:
        return 0  # hold


class RandomPolicy:
    """
    Baseline 2: fully random among {hold, buy YES, buy NO}.
    """
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> int:
        # action_space: 0=hold, 1=buy YES, 2=buy NO
        return int(self.rng.integers(0, 3))


class ProbHeuristicPolicy:
    """
    Baseline 3: simple probability-based heuristic.

    Uses:
      - yes_midprice = obs[6]
      - no_midprice  = obs[8]

    - If YES price < 0.45: buy YES  (underpriced YES)
    - If NO  price < 0.45: buy NO   (underpriced NO)
    - Else: hold
    """
    def select_action(self, obs: np.ndarray) -> int:
        yes_midprice = float(obs[6])
        no_midprice = float(obs[8])

        if yes_midprice < 0.45:
            return 1  # buy YES
        elif no_midprice < 0.45:
            return 2  # buy NO
        else:
            return 0  # hold
