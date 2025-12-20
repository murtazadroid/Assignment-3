# agent/train_multi_seed.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.data_loader import generate_synthetic_btc_minute_data, split_into_days
from env.kalshi_env import KalshiBTCHourlyEnv
from env.config import ENV_KWARGS


TOTAL_TIMESTEPS = 600_000
SEEDS = [0, 1, 2]


def make_env_fn(day_list, seed):
    def _init():
        env = KalshiBTCHourlyEnv(day_data_list=day_list, **ENV_KWARGS)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def train_one_seed(seed: int):
    print(f"\n===== Training PPO with seed={seed} =====")

    df = generate_synthetic_btc_minute_data(num_days=200)
    day_list = split_into_days(df)

    venv = DummyVecEnv([make_env_fn(day_list, seed)])
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.995,
    )

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        gamma=0.995,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1,
        seed=seed,
        device="auto",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"ppo_seed{seed}"
    vecnorm_path = models_dir / f"ppo_seed{seed}_vecnorm.pkl"

    model.save(str(model_path))
    venv.save(str(vecnorm_path))

    print(f"✅ Saved: {model_path}.zip")
    print(f"✅ Saved: {vecnorm_path}")


def main():
    for seed in SEEDS:
        train_one_seed(seed)


if __name__ == "__main__":
    main()
