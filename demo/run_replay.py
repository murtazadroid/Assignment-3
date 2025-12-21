# demo/run_replay.py

import csv
import sys
from pathlib import Path
import numpy as np

# Ensure project root on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env.data_loader import generate_synthetic_btc_minute_data, split_into_days
from env.kalshi_env import KalshiBTCHourlyEnv
from env.config import ENV_KWARGS


OBS_NAMES = [
    "r_5m", "r_15m", "r_60m", "vol_30m",
    "time_to_event", "moneyness",
    "yes_midprice_obs", "yes_spread_obs",
    "position", "cash_norm", "unrealized_norm",
]


def make_env_from_day_list(day_list, seed: int = 0):
    def _make_raw():
        e = KalshiBTCHourlyEnv(day_data_list=day_list, **ENV_KWARGS)
        e.reset(seed=seed)
        return e

    raw = _make_raw()
    venv = DummyVecEnv([lambda: Monitor(_make_raw())])
    return raw, venv


def run_episode(raw_env, venv, model, writer, episode_id: int):
    obs, _ = raw_env.reset()
    start = raw_env.start_balance
    step_idx = 0
    prev_pv = start

    while True:
        # Log raw obs features (not normalized) so we can plot volatility etc.
        obs_dict = {OBS_NAMES[i]: float(obs[i]) for i in range(min(len(obs), len(OBS_NAMES)))}

        # Model acts on normalized obs (VecNormalize stats)
        obs_norm = venv.normalize_obs(obs[None, :])
        action, _ = model.predict(obs_norm, deterministic=True)
        action = int(np.asarray(action).item())

        obs2, reward, term, trunc, info = raw_env.step(action)

        pv = float(info["portfolio_value"])
        dpv = pv - prev_pv
        prev_pv = pv

        row = {
            "episode": episode_id,
            "step": step_idx,
            "hour": info.get("hour"),
            "decision_price": info.get("decision_price"),
            "threshold": info.get("threshold"),
            "resolve_price": info.get("resolve_price"),
            "payoff_yes": info.get("payoff_yes"),
            "payoff_no": info.get("payoff_no"),
            "yes_midprice": info.get("yes_midprice"),
            "no_midprice": info.get("no_midprice"),
            "trade_side": info.get("trade_side"),
            "trade_cost": info.get("trade_cost"),
            "reward": float(reward),
            "portfolio_value": pv,
            "delta_pv": float(dpv),
        }
        row.update(obs_dict)

        writer.writerow(row)

        obs = obs2
        step_idx += 1

        if term or trunc:
            return pv - start


def main():
    # Default to local trained model artifacts
    MODEL_ZIP = ROOT / "models" / "ppo_best.zip"
    VECN_PKL  = ROOT / "models" / "ppo_best_vecnorm.pkl"

    if not MODEL_ZIP.exists() or not VECN_PKL.exists():
        raise FileNotFoundError(
            f"Missing model files: {MODEL_ZIP} and/or {VECN_PKL}. "
            "Train first (e.g., agent/train_rl.py) or point these paths to your artifacts."
        )

    model = PPO.load(str(MODEL_ZIP))

    # Synthetic data replay
    df = generate_synthetic_btc_minute_data(num_days=500, seed=123)
    day_list = split_into_days(df)

    raw_env, venv = make_env_from_day_list(day_list, seed=0)
    venv = VecNormalize.load(str(VECN_PKL), venv)
    venv.training = False
    venv.norm_reward = False

    out_dir = Path("demo_logs")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "replay_trades.csv"

    fields = [
        "episode","step","hour",
        "decision_price","threshold","resolve_price",
        "payoff_yes","payoff_no",
        "yes_midprice","no_midprice",
        "trade_side","trade_cost",
        "reward","portfolio_value","delta_pv",
        *OBS_NAMES
    ]

    N_EPISODES = 200
    pnls = []

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(N_EPISODES):
            pnl = run_episode(raw_env, venv, model, writer, ep)
            pnls.append(pnl)

    pnls = np.array(pnls, dtype=float)
    print("✅ Saved:", out_csv)
    print("episodes:", N_EPISODES)
    print("mean pnl:", float(pnls.mean()))
    print("std pnl :", float(pnls.std()))
    print("win rate:", float((pnls > 0).mean()))


if __name__ == "__main__":
    main()
