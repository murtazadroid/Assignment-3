# backtest/run_backtest_multiseed.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from env.data_loader import generate_synthetic_btc_minute_data, split_into_days
from env.kalshi_env import KalshiBTCHourlyEnv
from env.config import ENV_KWARGS
from backtest.baselines import RandomPolicy, ProbHeuristicPolicy


SEEDS = [0, 1, 2]
NUM_EPISODES = 100


def make_eval_env():
    # FIXED evaluation set (important!)
    df = generate_synthetic_btc_minute_data(num_days=200, seed=999)
    day_list = split_into_days(df)
    return KalshiBTCHourlyEnv(day_data_list=day_list, **ENV_KWARGS)


def evaluate_policy(env, policy):
    pnls = []
    start_balance = env.start_balance

    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        final_pv = start_balance

        while True:
            action = policy.select_action(obs)
            obs, _, term, trunc, info = env.step(action)
            final_pv = info["portfolio_value"]

            if term or trunc:
                pnls.append(final_pv - start_balance)
                break

    return np.array(pnls)


def evaluate_oracle(env):
    pnls = []
    start_balance = env.start_balance

    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        final_pv = start_balance

        while True:
            idx = env.current_step_idx
            thr = env.thresholds[idx]
            res = env.resolve_prices[idx]
            action = 1 if res > thr else 2

            obs, _, term, trunc, info = env.step(action)
            final_pv = info["portfolio_value"]

            if term or trunc:
                pnls.append(final_pv - start_balance)
                break

    return np.array(pnls)


def summarize(name, pnls):
    print(f"\n=== {name} ===")
    print(f"Mean PnL: {pnls.mean():.4f}")
    print(f"Std  PnL: {pnls.std():.4f}")
    print(f"Win rate: {(pnls > 0).mean():.2%}")


def main():
    env = make_eval_env()

    summarize("Random", evaluate_policy(env, RandomPolicy(seed=0)))
    summarize("Heuristic", evaluate_policy(env, ProbHeuristicPolicy()))
    summarize("Oracle", evaluate_oracle(env))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    all_seed_pnls = []

    for seed in SEEDS:
        print(f"\nEvaluating PPO seed {seed}")

        zip_path = Path("models") / f"ppo_seed{seed}.zip"
        vec_path = Path("models") / f"ppo_seed{seed}_vecnorm.pkl"

        def _init():
            return Monitor(make_eval_env())

        venv = DummyVecEnv([_init])
        venv = VecNormalize.load(str(vec_path), venv)
        venv.training = False
        venv.norm_reward = False

        model = PPO.load(str(zip_path))

        class PPOPolicy:
            def select_action(self, obs):
                obs_norm = venv.normalize_obs(obs[None, :])
                a, _ = model.predict(obs_norm, deterministic=True)
                return int(np.asarray(a).item())

        pnls = evaluate_policy(env, PPOPolicy())
        summarize(f"PPO_seed{seed}", pnls)
        all_seed_pnls.append(pnls.mean())

    print("\n=== PPO ROBUSTNESS SUMMARY ===")
    print(f"Mean across seeds: {np.mean(all_seed_pnls):.4f}")
    print(f"Std across seeds:  {np.std(all_seed_pnls):.4f}")


if __name__ == "__main__":
    main()
