# backtest/run_backtest.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from env.data_loader import (
    generate_synthetic_btc_minute_data,
    split_into_days,
)
from env.kalshi_env import KalshiBTCHourlyEnv
from backtest.baselines import NoTradePolicy, RandomPolicy, ProbHeuristicPolicy


def make_env(num_days: int = 200) -> KalshiBTCHourlyEnv:
    """
    Create a KalshiBTCHourlyEnv using synthetic BTC data for backtesting.
    Later we can swap this to real BTC data loading.
    """
    df = generate_synthetic_btc_minute_data(num_days=num_days)
    day_list = split_into_days(df)
    env = KalshiBTCHourlyEnv(
        day_data_list=day_list,
        start_balance=10_000.0,
        fee_per_contract=0.001,
        spread=0.02,
    )
    return env


def evaluate_policy(env: KalshiBTCHourlyEnv, policy, num_episodes: int = 100):
    """
    Run num_episodes episodes (days) with a given policy.

    Returns:
      - daily_rewards: array of total reward per episode
      - daily_pnls:    array of (final_portfolio_value - start_balance) per episode
    """
    daily_rewards = []
    daily_pnls = []

    start_balance = env.start_balance

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        final_pv = start_balance

        while True:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_pv = info.get("portfolio_value", final_pv)

            if terminated or truncated:
                daily_rewards.append(total_reward)
                daily_pnls.append(final_pv - start_balance)
                break

    return np.array(daily_rewards), np.array(daily_pnls)


def summarize_results(name: str, daily_pnls: np.ndarray, start_balance: float):
    """
    Print a summary of performance metrics for a given policy:
      - Mean / std daily PnL
      - Mean daily return
      - Daily and annualized Sharpe ratio
      - Min / Max daily PnL
      - Max drawdown (on cumulative equity)
      - Win rate (fraction of positive-PnL days)
    """
    if len(daily_pnls) == 0:
        print(f"\n=== Results for {name} ===")
        print("No PnL data available.")
        return

    # Convert PnL to daily returns (relative to starting balance)
    daily_returns = daily_pnls / start_balance

    mean_pnl = daily_pnls.mean()
    std_pnl = daily_pnls.std()

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    if std_ret > 1e-8:
        sharpe_daily = mean_ret / std_ret
        sharpe_annual = sharpe_daily * np.sqrt(252)  # approximate annualization
    else:
        sharpe_daily = np.nan
        sharpe_annual = np.nan

    # Build equity curve across episodes for max drawdown
    equity_curve = start_balance + np.cumsum(daily_pnls)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_dd = drawdowns.min()  # negative number, e.g. -0.25 = -25%

    # Win rate: fraction of days with positive PnL
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
    # 1. Build environment with synthetic data
    env = make_env(num_days=200)
    start_balance = env.start_balance

    # 2. Baseline policies
    no_trade = NoTradePolicy()
    random_pol = RandomPolicy(seed=0)
    prob_heuristic = ProbHeuristicPolicy()

    # 3. Evaluate each baseline
    print("Evaluating NoTradePolicy...")
    r_nt, pnl_nt = evaluate_policy(env, no_trade, num_episodes=100)
    summarize_results("NoTradePolicy", pnl_nt, start_balance)

    print("Evaluating RandomPolicy...")
    r_rand, pnl_rand = evaluate_policy(env, random_pol, num_episodes=100)
    summarize_results("RandomPolicy", pnl_rand, start_balance)

    print("Evaluating ProbHeuristicPolicy...")
    r_ph, pnl_ph = evaluate_policy(env, prob_heuristic, num_episodes=100)
    summarize_results("ProbHeuristicPolicy", pnl_ph, start_balance)

    # 4. (Optional) Evaluate trained PPO agent if model exists
    try:
        from stable_baselines3 import PPO

        print("\nEvaluating PPO trained agent (models/ppo_kalshi_test)...")

        model = PPO.load("models/ppo_kalshi_test")

        class PPOPolicyWrapper:
            def __init__(self, model):
                self.model = model

            def select_action(self, obs):
                action, _ = self.model.predict(obs, deterministic=True)
                return int(action)

        ppo_policy = PPOPolicyWrapper(model)
        r_ppo, pnl_ppo = evaluate_policy(env, ppo_policy, num_episodes=100)
        summarize_results("PPO_kalshi_test", pnl_ppo, start_balance)
    except Exception as e:
        print("\n[Info] Could not evaluate PPO model (maybe not trained yet):", e)


if __name__ == "__main__":
    main()
