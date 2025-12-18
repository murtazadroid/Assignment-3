# agent/train_rl.py

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from env.data_loader import (
    generate_synthetic_btc_minute_data,
    split_into_days,
)
from env.kalshi_env import KalshiBTCHourlyEnv


def make_env():
    """
    Helper to construct the KalshiBTCHourlyEnv with synthetic data.
    Later you can replace this with real BTC data loading.
    """
    # 1. Generate synthetic BTC data for, say, 30 days
    df = generate_synthetic_btc_minute_data(num_days=30)
    day_list = split_into_days(df)

    # 2. Create environment instance
    env = KalshiBTCHourlyEnv(
        day_data_list=day_list,
        start_balance=10_000.0,
        fee_per_contract=0.001,
        spread=0.02,
    )
    return env


def main():
    # Output directory for models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Create env
    env = make_env()

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,       # rollout length
        batch_size=256,
        verbose=1,
    )

    # Train for a small number of timesteps first (sanity check)
    total_timesteps = 20_000
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model_path = models_dir / "ppo_kalshi_test"
    model.save(model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
