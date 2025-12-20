# backtest/run_backtest.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from env.data_loader import generate_synthetic_btc_minute_data, split_into_days
from env.kalshi_env import KalshiBTCHourlyEnv
from env.config import ENV_KWARGS
from backtest.baselines import NoTradePolicy, RandomPolicy, ProbHeuristicPolicy


def make_env(num_days: int = 200) -> KalshiBTCHourlyEnv:
    df = generate_synthetic_btc_minute_data(num_days=num_days)
    day_list = split_into_days(df)
    return KalshiBTCHourlyEnv(day_data_list=day_list, **ENV_KWARGS)


def evaluate_policy(env: KalshiBTCHourlyEnv, policy, num_episodes: int = 100):
    daily_pnls = []
    start_balance = env.start_balance

    for _ in range(num_episodes):
        obs, _ = env.reset()
        final_pv = start_balance

        while True:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            final_pv = info.get("portfolio_value", final_pv)

            if terminated or truncated:
                daily_pnls.append(final_pv - start_balance)
                break

    return np.array(daily_pnls, dtype=np.float64)


def evaluate_oracle(env: KalshiBTCHourlyEnv, num_episodes: int = 100):
    daily_pnls = []
    start_balance = env.start_balance

    for _ in range(num_episodes):
        obs, _ = env.reset()
        final_pv = start_balance

        while True:
            idx = env.current_step_idx
            threshold = env.thresholds[idx]
            resolve_price = env.resolve_prices[idx]

            action = 1 if resolve_price > threshold else 2  # cheat
            obs, reward, terminated, truncated, info = env.step(action)
            final_pv = info.get("portfolio_value", final_pv)

            if terminated or truncated:
                daily_pnls.append(final_pv - start_balance)
                break

    return np.array(daily_pnls, dtype=np.float64)


def summarize_results(name: str, daily_pnls: np.ndarray, start_balance: float):
    if len(daily_pnls) == 0:
        print(f"\n=== Results for {name} ===\nNo data.")
        return

    daily_returns = daily_pnls / start_balance

    mean_pnl = daily_pnls.mean()
    std_pnl = daily_pnls.std()

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    if std_ret > 1e-8:
        sharpe_daily = mean_ret / std_ret
        sharpe_annual = sharpe_daily * np.sqrt(252)
    else:
        sharpe_daily = np.nan
        sharpe_annual = np.nan

    equity = start_balance + np.cumsum(daily_pnls)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    max_dd = drawdowns.min()

    win_rate = (daily_pnls > 0).mean()

    print(f"\n=== Results for {name} ===")
    print(f"Mean daily PnL:        {mean_pnl:.4f}")
    print(f"Std  daily PnL:        {std_pnl:.4f}")
    print(f"Mean daily return:     {mean_ret:.6f}")
    print(f"Sharpe (daily):        {sharpe_daily:.4f}")
    print(f"Sharpe (annualized):   {sharpe_annual:.4f}")
    print(f"Min / Max daily PnL:   {daily_pnls.min():.4f} / {daily_pnls.max():.4f}")
    print(f"Max drawdown:          {max_dd:.2%}")
    print(f"Win rate (days > 0):   {win_rate:.2%}")


def main():
    env = make_env(num_days=200)
    start_balance = env.start_balance

    print("Evaluating NoTradePolicy...")
    pnl_nt = evaluate_policy(env, NoTradePolicy())
    summarize_results("NoTradePolicy", pnl_nt, start_balance)

    print("Evaluating RandomPolicy...")
    pnl_rand = evaluate_policy(env, RandomPolicy(seed=0))
    summarize_results("RandomPolicy", pnl_rand, start_balance)

    print("Evaluating ProbHeuristicPolicy...")
    pnl_ph = evaluate_policy(env, ProbHeuristicPolicy())
    summarize_results("ProbHeuristicPolicy", pnl_ph, start_balance)

    print("Evaluating ORACLE (cheating diagnostic)...")
    pnl_oracle = evaluate_oracle(env)
    summarize_results("ORACLE", pnl_oracle, start_balance)

    # ---- PPO (loads ppo_latest ONLY) ----
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor

        models_dir = Path("models")
        zip_path = models_dir / "ppo_latest.zip"
        vecnorm_path = models_dir / "ppo_latest_vecnorm.pkl"

        print(f"\nEvaluating PPO trained agent ({zip_path})...")
        print("Model exists?", zip_path.exists(), "size:", zip_path.stat().st_size if zip_path.exists() else None)
        print("VecNormalize exists?", vecnorm_path.exists(), "size:", vecnorm_path.stat().st_size if vecnorm_path.exists() else None)

        # Create a VecEnv wrapper for normalization ONLY (we'll still step the raw env)
        def _init():
            return Monitor(make_env(num_days=200))

        venv = DummyVecEnv([_init])
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False

        model = PPO.load(str(zip_path))

        class PPOPolicyWrapper:
            def __init__(self, model, venv):
                self.model = model
                self.venv = venv

            def select_action(self, obs):
                obs_batched = obs[None, :]
                obs_norm = self.venv.normalize_obs(obs_batched)
                action, _ = self.model.predict(obs_norm, deterministic=True)
                return int(action)

        ppo_policy = PPOPolicyWrapper(model, venv)

        # debug action sample
        obs, _ = env.reset()
        acts = []
        for _ in range(15):
            a = ppo_policy.select_action(obs)
            acts.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        print("Sample PPO actions:", acts)

        pnl_ppo = evaluate_policy(env, ppo_policy)
        summarize_results("PPO_latest", pnl_ppo, start_balance)

    except Exception as e:
        print("\n[Info] PPO evaluation failed:", e)


if __name__ == "__main__":
    main()
