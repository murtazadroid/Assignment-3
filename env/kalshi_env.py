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
    - Each step = one hourly Kalshi-style event
    - Action space: 0=hold, 1=buy 1 YES, 2=sell 1 YES
    - Position is settled and reset to 0 after each event (no carryover)
    - Reward = change in portfolio value - small trading penalty

    NOTE (toy setup): threshold is generated using resolve_price as a center (future-leaky),
    but randomized so payoff is not always 1.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        day_data_list: List[pd.DataFrame],
        start_balance: float = 10_000.0,
        fee_per_contract: float = 0.001,
        spread: float = 0.02,
        threshold_band: float = 0.01,  # +/-1% around resolve price
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.day_data_list = day_data_list
        self.start_balance = start_balance
        self.fee_per_contract = fee_per_contract
        self.spread = spread
        self.threshold_band = threshold_band
        self.render_mode = render_mode

        # RNG (set in reset(seed=...))
        self.rng = np.random.default_rng()

        # Episode-level state
        self.current_day_idx: int = 0
        self.current_day_df: Optional[pd.DataFrame] = None

        # Hourly events from 9:00 to 23:00 inclusive
        self.event_hours = list(range(9, 24))
        self.num_events = len(self.event_hours)
        self.current_step_idx: int = 0

        # Portfolio state
        self.cash: float = self.start_balance
        self.position_yes: int = 0  # will be settled to 0 each step
        self.unrealized_pnl: float = 0.0
        self.prev_portfolio_value: float = self.start_balance

        # Precomputed per-event values
        self.thresholds: List[float] = []
        self.resolve_prices: List[float] = []

        # RL interface
        self.action_space = spaces.Discrete(3)

        # Observation: 11 features (floats)
        obs_low = np.array([-np.inf] * 11, dtype=np.float32)
        obs_high = np.array([np.inf] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helper: select day and precompute thresholds + resolve prices
    # ------------------------------------------------------------------
    def _reset_day(self):
        self.current_day_idx = self.rng.integers(0, len(self.day_data_list))
        self.current_day_df = self.day_data_list[self.current_day_idx]

        self.thresholds = []
        self.resolve_prices = []

        for hour in self.event_hours:
            resolve_time = self._get_timestamp_for_hour(hour)
            resolve_price = self._get_price_at_time(resolve_time)
            self.resolve_prices.append(resolve_price)

            # Toy threshold: resolve_price is "mid", choose random threshold around it
            # threshold = resolve_price * (1 + u), u ~ Uniform(-band, +band)
            u = self.rng.uniform(-self.threshold_band, self.threshold_band)
            threshold = resolve_price * (1.0 + u)
            self.thresholds.append(float(threshold))

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
    # Gym API: reset & step
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
        info = {"portfolio_value": self.prev_portfolio_value}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Current event details
        hour = self.event_hours[self.current_step_idx]
        threshold = self.thresholds[self.current_step_idx]

        # Decision time = 5 minutes before the hour
        decision_time = self._get_timestamp_for_hour(hour) - pd.Timedelta(minutes=5)
        current_price = self._get_price_at_time(decision_time)

        # YES contract price (mid) based on moneyness
        yes_midprice = self._simulate_yes_midprice(current_price, threshold)
        bid, ask = self._get_bid_ask(yes_midprice)

        # --- Execute action (no shorting) ---
        trade_size = 0
        if action == 1:  # buy 1 YES at ask
            cost = ask + self.fee_per_contract
            if self.cash >= cost:
                self.cash -= cost
                self.position_yes += 1
                trade_size = 1
        elif action == 2:  # sell 1 YES at bid (only if holding)
            if self.position_yes >= 1:
                self.cash += bid - self.fee_per_contract
                self.position_yes -= 1
                trade_size = -1

        # --- Resolve event at the hour ---
        resolve_price = self.resolve_prices[self.current_step_idx]
        payoff_yes = 1.0 if resolve_price > threshold else 0.0

        # --- Settle & clear position after each event ---
        # Any remaining YES position resolves to cash immediately.
        if self.position_yes != 0:
            self.cash += self.position_yes * payoff_yes
            self.position_yes = 0

        portfolio_value = self.cash

        # Reward = change in portfolio value - small penalty for trading
        trade_penalty = 0.001 * abs(trade_size)
        reward = (portfolio_value - self.prev_portfolio_value) - trade_penalty
        self.prev_portfolio_value = portfolio_value
        self.unrealized_pnl = 0.0

        # Move to next hour
        self.current_step_idx += 1
        terminated = self.current_step_idx >= self.num_events
        truncated = False

        obs = self._get_observation()
        info = {
            "hour": hour,
            "threshold": float(threshold),
            "resolve_price": float(resolve_price),
            "payoff_yes": float(payoff_yes),
            "yes_midprice": float(yes_midprice),
            "bid": float(bid),
            "ask": float(ask),
            "trade_size": int(trade_size),
            "portfolio_value": float(portfolio_value),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        # Use a safe index when episode is done
        hour_idx = min(self.current_step_idx, self.num_events - 1)
        hour = self.event_hours[hour_idx]
        threshold = self.thresholds[hour_idx]

        decision_time = self._get_timestamp_for_hour(hour) - pd.Timedelta(minutes=5)
        current_price = self._get_price_at_time(decision_time)

        r_5m, r_15m, r_60m, vol_30m = self._compute_price_features(decision_time)

        time_to_event = 5.0 / 60.0  # always decide 5 min before
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
        prob = 1.0 / (1.0 + np.exp(-5.0 * m))  # logistic
        return float(prob)

    def _get_bid_ask(self, midprice: float):
        half_spread = self.spread / 2.0
        bid = np.clip(midprice - half_spread, 0.0, 1.0)
        ask = np.clip(midprice + half_spread, 0.0, 1.0)
        return float(bid), float(ask)

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            print(
                f"Step {self.current_step_idx}, cash={self.cash:.4f}, pos_yes={self.position_yes}"
            )

    def close(self):
        pass
