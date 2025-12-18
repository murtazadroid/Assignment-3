# env/kalshi_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional


class KalshiBTCHourlyEnv(gym.Env):
    """
    Simplified Gym environment for BTC hourly threshold events.

    - One episode = one day (9:00 to 23:00)
    - Each step = one hourly event
    - Action space: 0=hold, 1=buy 1 YES, 2=sell 1 YES
    - Position settles and resets to 0 after each event (no carryover)
    - Reward = change in portfolio value - small trading penalty

    IMPORTANT FIX:
    Threshold (strike) is computed using decision-time price (5 min before the hour),
    snapped to a strike grid (e.g., $50 or $100 increments). No future leakage.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        day_data_list: List[pd.DataFrame],
        start_balance: float = 10_000.0,
        fee_per_contract: float = 0.001,
        spread: float = 0.02,
        strike_step: float = 100.0,   # strike grid size (use 50.0 or 100.0)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.day_data_list = day_data_list
        self.start_balance = start_balance
        self.fee_per_contract = fee_per_contract
        self.spread = spread
        self.strike_step = float(strike_step)
        self.render_mode = render_mode

        # RNG (set in reset(seed=...))
        self.rng = np.random.default_rng()

        # Episode state
        self.current_day_idx: int = 0
        self.current_day_df: Optional[pd.DataFrame] = None

        # Hourly events from 9:00 to 23:00 inclusive
        self.event_hours = list(range(9, 24))
        self.num_events = len(self.event_hours)
        self.current_step_idx: int = 0

        # Portfolio state
        self.cash: float = self.start_balance
        self.position_yes: int = 0
        self.unrealized_pnl: float = 0.0
        self.prev_portfolio_value: float = self.start_balance

        # Per-event values
        self.thresholds: List[float] = []
        self.resolve_prices: List[float] = []
        self.decision_prices: List[float] = []  # for debug/analysis

        # RL interface
        self.action_space = spaces.Discrete(3)

        # Observation: 11 floats
        obs_low = np.array([-np.inf] * 11, dtype=np.float32)
        obs_high = np.array([np.inf] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reset_day(self):
        self.current_day_idx = int(self.rng.integers(0, len(self.day_data_list)))
        self.current_day_df = self.day_data_list[self.current_day_idx]

        self.thresholds = []
        self.resolve_prices = []
        self.decision_prices = []

        for hour in self.event_hours:
            resolve_time = self._get_timestamp_for_hour(hour)
            decision_time = resolve_time - pd.Timedelta(minutes=5)

            decision_price = self._get_price_at_time(decision_time)
            resolve_price = self._get_price_at_time(resolve_time)

            # Strike grid: snap to nearest strike_step around decision price
            # Example: if strike_step=100 and price=43127 -> strike=43100
            strike = round(decision_price / self.strike_step) * self.strike_step

            self.decision_prices.append(float(decision_price))
            self.resolve_prices.append(float(resolve_price))
            self.thresholds.append(float(strike))

    def _get_timestamp_for_hour(self, hour: int) -> pd.Timestamp:
        date = self.current_day_df.index[0].date()
        ts = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=0,
            tz=self.current_day_df.index.tz,
        )
        return ts

    def _get_price_at_time(self, ts: pd.Timestamp) -> float:
        df = self.current_day_df
        if ts in df.index:
            return float(df.loc[ts, "close"])
        before = df[df.index <= ts]
        if len(before) == 0:
            return float(df["close"].iloc[0])
        return float(before["close"].iloc[-1])

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_day()
        self.current_step_idx = 0

        self.cash = self.start_balance
        self.position_yes = 0
        self.unrealized_pnl = 0.0
        self.prev_portfolio_value = self.start_balance

        obs = self._get_observation()
        info = {"portfolio_value": float(self.prev_portfolio_value)}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        hour = self.event_hours[self.current_step_idx]
        threshold = self.thresholds[self.current_step_idx]

        resolve_time = self._get_timestamp_for_hour(hour)
        decision_time = resolve_time - pd.Timedelta(minutes=5)

        current_price = self._get_price_at_time(decision_time)

        # Simulated YES price based on moneyness at decision time
        yes_midprice = self._simulate_yes_midprice(current_price, threshold)
        bid, ask = self._get_bid_ask(yes_midprice)

        # --- Execute action (no shorting) ---
        trade_size = 0
        if action == 1:  # buy 1 YES
            cost = ask + self.fee_per_contract
            if self.cash >= cost:
                self.cash -= cost
                self.position_yes += 1
                trade_size = 1
        elif action == 2:  # sell 1 YES (only if holding)
            if self.position_yes >= 1:
                self.cash += bid - self.fee_per_contract
                self.position_yes -= 1
                trade_size = -1

        # --- Resolve at the hour ---
        resolve_price = self.resolve_prices[self.current_step_idx]
        payoff_yes = 1.0 if resolve_price > threshold else 0.0

        # --- Settle and clear position after each event ---
        if self.position_yes != 0:
            self.cash += self.position_yes * payoff_yes
            self.position_yes = 0

        portfolio_value = self.cash

        # Reward = delta PV - small trading penalty
        trade_penalty = 0.001 * abs(trade_size)
        reward = (portfolio_value - self.prev_portfolio_value) - trade_penalty
        self.prev_portfolio_value = portfolio_value
        self.unrealized_pnl = 0.0

        # Advance
        self.current_step_idx += 1
        terminated = self.current_step_idx >= self.num_events
        truncated = False

        obs = self._get_observation()
        info = {
            "hour": int(hour),
            "threshold": float(threshold),
            "decision_price": float(current_price),
            "resolve_price": float(resolve_price),
            "payoff_yes": float(payoff_yes),
            "yes_midprice": float(yes_midprice),
            "bid": float(bid),
            "ask": float(ask),
            "trade_size": int(trade_size),
            "portfolio_value": float(portfolio_value),
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step_idx, self.num_events - 1)

        hour = self.event_hours[idx]
        threshold = self.thresholds[idx]

        resolve_time = self._get_timestamp_for_hour(hour)
        decision_time = resolve_time - pd.Timedelta(minutes=5)
        current_price = self._get_price_at_time(decision_time)

        r_5m, r_15m, r_60m, vol_30m = self._compute_price_features(decision_time)

        time_to_event = 5.0 / 60.0
        moneyness = (current_price - threshold) / max(threshold, 1e-6)

        yes_midprice = self._simulate_yes_midprice(current_price, threshold)
        bid, ask = self._get_bid_ask(yes_midprice)
        yes_spread = ask - bid

        cash_norm = self.cash / self.start_balance
        unrealized_norm = self.unrealized_pnl / self.start_balance

        obs = np.array(
            [
                r_5m,
                r_15m,
                r_60m,
                vol_30m,
                time_to_event,
                moneyness,
                yes_midprice,
                yes_spread,
                float(self.position_yes),
                cash_norm,
                unrealized_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_price_features(self, ts: pd.Timestamp):
        df = self.current_day_df
        before = df[df.index <= ts]
        if len(before) < 60:
            return 0.0, 0.0, 0.0, 0.0

        def ret(minutes: int) -> float:
            window = before.iloc[-minutes:]
            return float(window["close"].iloc[-1] / window["close"].iloc[0] - 1.0)

        r_5m = ret(5)
        r_15m = ret(15)
        r_60m = ret(60)

        window_30 = before.iloc[-30:]
        log_returns = np.log(window_30["close"]).diff().dropna()
        vol_30m = float(log_returns.std())

        return r_5m, r_15m, r_60m, vol_30m

    def _simulate_yes_midprice(self, current_price: float, threshold: float) -> float:
        m = (current_price - threshold) / max(threshold, 1e-6)
        prob = 1.0 / (1.0 + np.exp(-5.0 * m))
        return float(prob)

    def _get_bid_ask(self, midprice: float):
        half_spread = self.spread / 2.0
        bid = np.clip(midprice - half_spread, 0.0, 1.0)
        ask = np.clip(midprice + half_spread, 0.0, 1.0)
        return float(bid), float(ask)

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step_idx}, cash={self.cash:.4f}, pos_yes={self.position_yes}")

    def close(self):
        pass
