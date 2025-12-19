# env/kalshi_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional


class KalshiBTCHourlyEnv(gym.Env):
    """
    BTC hourly prediction-market environment (Kalshi-style), one-shot settlement.

    One episode = one day (09:00–23:00)
    One step     = one hourly event

    Actions:
      0 = hold
      1 = buy 1 YES (one-shot)
      2 = buy 1 NO  (one-shot)

    Key design:
    - Decision time is `decision_offset_minutes` BEFORE resolve time.
    - Strike (threshold) is snapped to a strike grid based on decision-time price (no future leakage).
    - Market pricing is a toy probability model with optional bias/noise (mispricing) so RL can learn an edge.
    - Contracts settle immediately at resolve within the same step (no carry positions).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        day_data_list: List[pd.DataFrame],
        start_balance: float = 10_000.0,
        fee_per_contract: float = 0.001,
        spread: float = 0.02,
        strike_step: float = 100.0,
        decision_offset_minutes: int = 60,
        trade_penalty_coef: float = 0.001,
        pricing_k: float = 5.0,
        pricing_bias: float = 0.0,
        pricing_noise_std: float = 0.03,
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
        self.event_hours = list(range(9, 24))  # 9..23 inclusive
        self.num_events = len(self.event_hours)

        # Portfolio state (cash only; one-shot settlement)
        self.cash = self.start_balance
        self.prev_portfolio_value = self.start_balance

        # Per-event values
        self.thresholds: List[float] = []
        self.resolve_prices: List[float] = []

        # RL interface
        self.action_space = spaces.Discrete(3)

        # Observation: 11 features
        # [r5, r15, r60, vol30, time_to_event_hours, moneyness,
        #  yes_mid, yes_spread, no_mid, no_spread, cash_norm]
        obs_low = np.array([-np.inf] * 11, dtype=np.float32)
        obs_high = np.array([np.inf] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Time/price helpers
    # ------------------------------------------------------------------
    def _get_timestamp(self, hour: int) -> pd.Timestamp:
        d = self.current_day_df.index[0].date()
        return pd.Timestamp(d.year, d.month, d.day, hour, 0, tz=self.current_day_df.index.tz)

    def _get_price_at_time(self, ts: pd.Timestamp) -> float:
        df = self.current_day_df
        if ts in df.index:
            return float(df.loc[ts, "close"])
        before = df[df.index <= ts]
        if len(before) == 0:
            return float(df["close"].iloc[0])
        return float(before["close"].iloc[-1])

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------
    def _reset_day(self):
        self.current_day_df = self.day_data_list[int(self.rng.integers(0, len(self.day_data_list)))]

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
        self.prev_portfolio_value = self.start_balance

        obs = self._get_observation()
        info = {"portfolio_value": float(self.prev_portfolio_value)}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        hour = self.event_hours[self.current_step_idx]
        threshold = self.thresholds[self.current_step_idx]

        resolve_time = self._get_timestamp(hour)
        decision_time = resolve_time - pd.Timedelta(minutes=self.decision_offset_minutes)
        decision_price = self._get_price_at_time(decision_time)

        # Market pricing (YES as probability with mispricing)
        yes_mid = self._yes_midprice(decision_price, threshold)
        yes_bid, yes_ask = self._bid_ask(yes_mid)

        # NO is complement (Kalshi-style)
        no_mid = 1.0 - yes_mid
        no_bid, no_ask = self._bid_ask(no_mid)

        # One-shot trade bookkeeping
        trade_side = 0  # 0=none, +1=YES, -1=NO
        trade_cost = 0.0

        # Execute action (buy only; settles at resolve within step)
        if action == 1:  # buy YES
            cost = yes_ask + self.fee_per_contract
            if self.cash >= cost:
                self.cash -= cost
                trade_side = +1
                trade_cost = cost

        elif action == 2:  # buy NO
            cost = no_ask + self.fee_per_contract
            if self.cash >= cost:
                self.cash -= cost
                trade_side = -1
                trade_cost = cost

        # Resolve event
        resolve_price = self.resolve_prices[self.current_step_idx]
        payoff_yes = 1.0 if resolve_price > threshold else 0.0
        payoff_no = 1.0 - payoff_yes

        # Settle immediately (no carry positions)
        if trade_side == +1:
            self.cash += payoff_yes
        elif trade_side == -1:
            self.cash += payoff_no

        portfolio_value = self.cash

        trade_penalty = self.trade_penalty_coef * (1.0 if action != 0 else 0.0)
        reward = (portfolio_value - self.prev_portfolio_value) - trade_penalty
        self.prev_portfolio_value = portfolio_value

        # Advance time
        self.current_step_idx += 1
        terminated = self.current_step_idx >= self.num_events
        truncated = False

        obs = self._get_observation()

        info = {
            "hour": int(hour),
            "decision_price": float(decision_price),
            "threshold": float(threshold),
            "resolve_price": float(resolve_price),
            "payoff_yes": float(payoff_yes),
            "payoff_no": float(payoff_no),
            "yes_midprice": float(yes_mid),
            "yes_bid": float(yes_bid),
            "yes_ask": float(yes_ask),
            "no_midprice": float(no_mid),
            "no_bid": float(no_bid),
            "no_ask": float(no_ask),
            "trade_side": int(trade_side),
            "trade_cost": float(trade_cost),
            "portfolio_value": float(portfolio_value),
        }

        return obs, float(reward), terminated, truncated, info

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

        time_to_event_hours = float(self.decision_offset_minutes) / 60.0
        moneyness = (price - threshold) / max(threshold, 1e-6)

        yes_mid = self._yes_midprice(price, threshold)
        yes_bid, yes_ask = self._bid_ask(yes_mid)
        yes_spread = yes_ask - yes_bid

        no_mid = 1.0 - yes_mid
        no_bid, no_ask = self._bid_ask(no_mid)
        no_spread = no_ask - no_bid

        cash_norm = self.cash / self.start_balance

        obs = np.array(
            [
                r5,
                r15,
                r60,
                vol30,
                time_to_event_hours,
                moneyness,
                yes_mid,
                yes_spread,
                no_mid,
                no_spread,
                cash_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def _price_features(self, ts: pd.Timestamp):
        df = self.current_day_df
        before = df[df.index <= ts]
        if len(before) < 60:
            return 0.0, 0.0, 0.0, 0.0

        def ret(m: int) -> float:
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
    # Toy market pricing
    # ------------------------------------------------------------------
    def _yes_midprice(self, price: float, threshold: float) -> float:
        """
        Toy market midprice for YES:
          prob = sigmoid(k * moneyness) + bias + noise
        """
        m = (price - threshold) / max(threshold, 1e-6)
        prob = 1.0 / (1.0 + np.exp(-self.pricing_k * m))
        prob = prob + self.pricing_bias + self.rng.normal(0.0, self.pricing_noise_std)
        return float(np.clip(prob, 0.001, 0.999))

    def _bid_ask(self, mid: float):
        half = self.spread / 2.0
        bid = float(np.clip(mid - half, 0.0, 1.0))
        ask = float(np.clip(mid + half, 0.0, 1.0))
        return bid, ask

    def render(self):
        if self.render_mode == "human":
            print(f"step={self.current_step_idx}, cash={self.cash:.4f}")

    def close(self):
        pass
