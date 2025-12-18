# env/kalshi_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional


class KalshiBTCHourlyEnv(gym.Env):
    """
    BTC hourly prediction-market environment (Kalshi-style).

    One episode = one trading day (09:00–23:00)
    One step     = one hourly event

    Actions:
      0 = hold
      1 = buy 1 YES
      2 = sell 1 YES (only if holding)

    IMPORTANT:
    - Threshold (strike) is computed from DECISION-TIME price
      and snapped to a strike grid (no future leakage).
    - Position settles and resets to 0 every step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        day_data_list: List[pd.DataFrame],
        start_balance: float = 10_000.0,
        fee_per_contract: float = 0.001,
        spread: float = 0.02,
        strike_step: float = 100.0,  # e.g. 50 or 100 USD
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.day_data_list = day_data_list
        self.start_balance = start_balance
        self.fee_per_contract = fee_per_contract
        self.spread = spread
        self.strike_step = float(strike_step)
        self.render_mode = render_mode

        self.rng = np.random.default_rng()

        # Episode state
        self.current_day_df = None
        self.current_step_idx = 0
        self.event_hours = list(range(9, 24))
        self.num_events = len(self.event_hours)

        # Portfolio
        self.cash = start_balance
        self.position_yes = 0
        self.prev_portfolio_value = start_balance

        # Per-event data
        self.thresholds = []
        self.resolve_prices = []

        # RL spaces
        self.action_space = spaces.Discrete(3)

        obs_low = np.array([-np.inf] * 11, dtype=np.float32)
        obs_high = np.array([np.inf] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reset_day(self):
        self.current_day_df = self.day_data_list[
            self.rng.integers(0, len(self.day_data_list))
        ]

        self.thresholds = []
        self.resolve_prices = []

        for hour in self.event_hours:
            resolve_time = self._get_timestamp(hour)
            decision_time = resolve_time - pd.Timedelta(minutes=5)

            decision_price = self._get_price_at_time(decision_time)
            resolve_price = self._get_price_at_time(resolve_time)

            strike = round(decision_price / self.strike_step) * self.strike_step

            self.thresholds.append(float(strike))
            self.resolve_prices.append(float(resolve_price))

    def _get_timestamp(self, hour: int) -> pd.Timestamp:
        d = self.current_day_df.index[0].date()
        return pd.Timestamp(d.year, d.month, d.day, hour, 0, tz=self.current_day_df.index.tz)

    def _get_price_at_time(self, ts: pd.Timestamp) -> float:
        df = self.current_day_df
        if ts in df.index:
            return float(df.loc[ts, "close"])
        before = df[df.index <= ts]
        return float(before["close"].iloc[-1]) if len(before) else float(df["close"].iloc[0])

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_day()
        self.current_step_idx = 0

        self.cash = self.start_balance
        self.position_yes = 0
        self.prev_portfolio_value = self.start_balance

        return self._get_observation(), {"portfolio_value": self.start_balance}

    def step(self, action: int):
        assert self.action_space.contains(action)

        hour = self.event_hours[self.current_step_idx]
        threshold = self.thresholds[self.current_step_idx]

        resolve_time = self._get_timestamp(hour)
        decision_time = resolve_time - pd.Timedelta(minutes=5)
        decision_price = self._get_price_at_time(decision_time)

        yes_mid = self._yes_midprice(decision_price, threshold)
        bid, ask = self._bid_ask(yes_mid)

        trade_size = 0

        # Execute action
        if action == 1:  # buy YES
            cost = ask + self.fee_per_contract
            if self.cash >= cost:
                self.cash -= cost
                self.position_yes = 1
                trade_size = 1

        elif action == 2 and self.position_yes == 1:  # sell YES
            self.cash += bid - self.fee_per_contract
            self.position_yes = 0
            trade_size = -1

        # Resolve event
        resolve_price = self.resolve_prices[self.current_step_idx]
        payoff_yes = 1.0 if resolve_price > threshold else 0.0

        if self.position_yes == 1:
            self.cash += payoff_yes
            self.position_yes = 0

        portfolio_value = self.cash
        reward = (portfolio_value - self.prev_portfolio_value) - 0.001 * abs(trade_size)
        self.prev_portfolio_value = portfolio_value

        self.current_step_idx += 1
        terminated = self.current_step_idx >= self.num_events

        obs = self._get_observation()
        info = {
            "hour": hour,
            "decision_price": decision_price,
            "threshold": threshold,
            "resolve_price": resolve_price,
            "payoff_yes": payoff_yes,
            "portfolio_value": portfolio_value,
        }

        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self):
        idx = min(self.current_step_idx, self.num_events - 1)
        hour = self.event_hours[idx]
        threshold = self.thresholds[idx]

        resolve_time = self._get_timestamp(hour)
        decision_time = resolve_time - pd.Timedelta(minutes=5)
        price = self._get_price_at_time(decision_time)

        r5, r15, r60, vol30 = self._price_features(decision_time)

        moneyness = (price - threshold) / max(threshold, 1e-6)
        yes_mid = self._yes_midprice(price, threshold)
        bid, ask = self._bid_ask(yes_mid)

        return np.array([
            r5, r15, r60, vol30,
            5.0 / 60.0,
            moneyness,
            yes_mid,
            ask - bid,
            float(self.position_yes),
            self.cash / self.start_balance,
            0.0,
        ], dtype=np.float32)

    def _price_features(self, ts):
        df = self.current_day_df
        before = df[df.index <= ts]
        if len(before) < 60:
            return 0, 0, 0, 0

        def ret(m):
            w = before.iloc[-m:]
            return float(w["close"].iloc[-1] / w["close"].iloc[0] - 1)

        r5 = ret(5)
        r15 = ret(15)
        r60 = ret(60)

        lr = np.log(before["close"]).diff().dropna()
        vol30 = float(lr.iloc[-30:].std()) if len(lr) >= 30 else 0.0

        return r5, r15, r60, vol30

    def _yes_midprice(self, price, threshold):
        m = (price - threshold) / max(threshold, 1e-6)
        return float(1 / (1 + np.exp(-5 * m)))

    def _bid_ask(self, mid):
        h = self.spread / 2
        return max(0, mid - h), min(1, mid + h)

    def render(self):
        if self.render_mode == "human":
            print(f"step={self.current_step_idx} cash={self.cash:.2f}")
