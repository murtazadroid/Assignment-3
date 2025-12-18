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
    Baseline 2: fully random among {hold, buy, sell}.
    """
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> int:
        # action_space: 0=hold, 1=buy YES, 2=sell YES
        return int(self.rng.integers(0, 3))


class ProbHeuristicPolicy:
    """
    Baseline 3: simple probability-based heuristic.

    Uses yes_midprice (obs[6]):

    - If price < 0.45: buy YES  (we think it's underpriced)
    - If price > 0.55: sell YES (we think it's overpriced)
    - Else: do nothing
    """
    def select_action(self, obs: np.ndarray) -> int:
        yes_midprice = float(obs[6])

        if yes_midprice < 0.45:
            return 1  # buy YES
        elif yes_midprice > 0.55:
            return 2  # sell YES
        else:
            return 0  # hold
