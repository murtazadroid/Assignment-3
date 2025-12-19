# env/kalshi_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional


class KalshiBTCHourlyEnv(gym.Env):
    """
    BTC hourly prediction-market environment (Kalshi-style).

    One episode = one day (09:00–23:00)
    One step     = one hourly event

    Actions:
      0 = hold
      1 = buy 1 YES
      2 = sell 1 YES (only if holding)

    Key design choices:
    - Strike (threshold) is set from DECISION-time price on a strike grid (no future leakage).
    - Decision happens `decision_offset_minutes` before resolve (default 60 minutes).
    - Market price (YES mid) is a *toy* function of moneyness, with optional mispricing noise/bias.
    - Position settles and resets to 0 after each event.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        day_data_list: List[pd.DataFrame],
        start_balance: float = 10_000.0,
        fee_per_contract: float = 0.001,
        spread: float = 0.02,
        strike_step: float = 100.0,              # e.g. 50 or 100 USD
        decision_offset_minutes: int = 60,       # decide 60 min before the hour resolves
        trade_penalty_coef: float = 0.001,       # extra penalty per trade to discourage churn
        pricing_k: float = 5.0,                  # logistic slope for toy midprice
        pricing_bias: float = 0.0,               # systematic mispricing shift (+ favors YES)
        pricing_noise_std: float = 0.03,         # random midprice noise (mispricing)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.day_data_list = day_data_list
        self.start_balance = float(start_balance)
        self.fee_per_contract = float(fee_per_contract)
        self.spread = float(spread)
        self.strike_step = float(strike_step)
        self.decision_offset_minutes = int(decision_offset_minutes)
        self.trade_penalty_coef = float(trade_penalty_coef)

        self.pricing_k = float(pricing_k)
        self.pricing_bias = float(pricing_bias)
        self.pricing_noise_std = float(pricing_noise_std)

        self.render_mode = render_mode
        self.rng = np.random.default_rng()

        # Episode state
        self.current_day_df: Optional[pd.DataFrame] = None
        self.current_step_idx: int = 0
        self.event_hours = list(range(9, 24))
        self.num_events = len(self.event_hours)

        # Portfolio (cash only, because we settle each event)
        self.cash = self.start_balance
        self.position_yes = 0
        self.prev_portfolio_value = self.start_balance

        # Per-event
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
    def _get_timestamp(self, hour: int) -> pd.Timestamp:
        d = self.current_day_df.index[0].date()
        return pd.Timestamp(
            d.year, d.month, d.day, hour, 0, tz=self.current_day_df.index.tz
        )

    def _get_price_at_time(self, ts: pd.Timestamp) -> float:
        df = self.current_day_df
        if ts in df.index:
            return float(df.loc[ts, "close"])
        before = df[df.index <= ts]
        if len(before) == 0:
            return float(df["close"].iloc[0])
        return float(before["close"].iloc[-1])

    def _reset_day(self):
        self.current_day_df = self.day_data_list[
            int(self.rng.integers(0, len(self.day_data_list)))
        ]

        self.thresholds = []
        self.resolve_prices = []

        for hour in self.event_hours:
            resolve_time = self._get_timestamp(hour)
            decision_time = resolve_time - pd.Timedelta(minutes=self.decision_offset_minutes)

            decision_price = self._get_price_at_time(decision_time)
            resolve_price = self._get_price_at_time(resolve_time)

            # Strike grid based on decision-time price (no leakage)
            strike = round(decision_price / self.strike_step) * self.strike_step

            self.thresholds.append(float(strike))
            self.resolve_prices.append(float(resolve_price))

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

        return self._get_observation(), {"portfolio_value": float(self.start_balance)}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        hour = self.event_hours[self.current_step_idx]
        threshold = self.thresholds[self.current_step_idx]

        resolve_time = self._get_timestamp(hour)
        decision_time = resolve_time - pd.Timedelta(minutes=self.decision_offset_minutes)

        decision_price = self._get_price_at_time(decision_time)

        # Toy market midprice (intentionally imperfect)
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

        elif action == 2:  # sell YES (only if holding)
            if self.position_yes == 1:
                self.cash += bid - self.fee_per_contract
                self.position_yes = 0
                trade_size = -1

        # Resolve event
        resolve_price = self.resolve_prices[self.current_step_idx]
        payoff_yes = 1.0 if resolve_price > threshold else 0.0

        # Settle and clear position (no carry)
        if self.position_yes == 1:
            self.cash += payoff_yes
            self.position_yes = 0

        portfolio_value = self.cash

        trade_penalty = self.trade_penalty_coef * abs(trade_size)
        reward = (portfolio_value - self.prev_portfolio_value) - trade_penalty
        self.prev_portfolio_value = portfolio_value

        self.current_step_idx += 1
        terminated = self.current_step_idx >= self.num_events

        obs = self._get_observation()
        info = {
            "hour": int(hour),
            "decision_price": float(decision_price),
            "threshold": float(threshold),
            "resolve_price": float(resolve_price),
            "payoff_yes": float(payoff_yes),
            "yes_midprice": float(yes_mid),
            "bid": float(bid),
            "ask": float(ask),
            "trade_size": int(trade_size),
            "portfolio_value": float(portfolio_value),
        }
        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step_idx, self.num_events - 1)

        hour = self.event_hours[idx]
        threshold = self.thresholds[idx]

        resolve_time = self._get_timestamp(hour)
        decision_time = resolve_time - pd.Timedelta(minutes=self.decision_offset_minutes)
        price = self._get_price_at_time(decision_time)

        r5, r15, r60, vol30 = self._price_features(decision_time)

        time_to_event = float(self.decision_offset_minutes) / 60.0
        moneyness = (price - threshold) / max(threshold, 1e-6)

        yes_mid = self._yes_midprice(price, threshold)
        bid, ask = self._bid_ask(yes_mid)

        obs = np.array(
            [
                r5,
                r15,
                r60,
                vol30,
                time_to_event,
                moneyness,
                yes_mid,
                (ask - bid),
                float(self.position_yes),
                float(self.cash / self.start_balance),
                0.0,  # unrealized pnl (unused because we settle each step)
            ],
            dtype=np.float32,
        )
        return obs

    def _price_features(self, ts: pd.Timestamp):
        df = self.current_day_df
        before = df[df.index <= ts]
        if len(before) < 60:
            return 0.0, 0.0, 0.0, 0.0

        def ret(m):
            w = before.iloc[-m:]
            return float(w["close"].iloc[-1] / w["close"].iloc[0] - 1.0)

        r5 = ret(5)
        r15 = ret(15)
        r60 = ret(60)

        window_30 = before.iloc[-30:]
        log_returns = np.log(window_30["close"]).diff().dropna()
        vol30 = float(log_returns.std()) if len(log_returns) else 0.0

        return r5, r15, r60, vol30

    # ------------------------------------------------------------------
    # Pricing model (toy market)
    # ------------------------------------------------------------------
    def _yes_midprice(self, price: float, threshold: float) -> float:
        """
        Toy "market" midprice.
        We intentionally allow mispricing so the agent can learn an edge.
        """
        m = (price - threshold) / max(threshold, 1e-6)
        prob = 1.0 / (1.0 + np.exp(-self.pricing_k * m))

        # systematic bias + random noise (mispricing)
        prob = prob + self.pricing_bias + self.rng.normal(0.0, self.pricing_noise_std)

        return float(np.clip(prob, 0.001, 0.999))

    def _bid_ask(self, mid: float):
        h = self.spread / 2.0
        bid = float(np.clip(mid - h, 0.0, 1.0))
        ask = float(np.clip(mid + h, 0.0, 1.0))
        return bid, ask

    def render(self):
        if self.render_mode == "human":
            print(f"step={self.current_step_idx} cash={self.cash:.4f}")

    def close(self):
        pass
