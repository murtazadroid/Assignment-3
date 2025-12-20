# agent/train_rl.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.data_loader import generate_synthetic_btc_minute_data, split_into_days
from env.kalshi_env import KalshiBTCHourlyEnv
from env.config import ENV_KWARGS

SEED = 123


def make_env_fn(num_days: int = 200, seed: int = SEED):
    """
    SB3 VecEnv needs a callable that returns an env.
    We build a fixed synthetic dataset and reuse it.
    """
    df = generate_synthetic_btc_minute_data(num_days=num_days)
    day_list = split_into_days(df)

    def _init():
        env = KalshiBTCHourlyEnv(day_data_list=day_list, **ENV_KWARGS)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Always save to these names so backtest can never load the wrong model
    RUN_NAME = "ppo_latest"
    model_path = models_dir / RUN_NAME
    vecnorm_path = models_dir / f"{RUN_NAME}_vecnorm.pkl"

    # Vec env + normalization (important when rewards are small/noisy)
    venv = DummyVecEnv([make_env_fn(num_days=200, seed=SEED)])
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
        ent_coef=0.05,  # ✅ stronger exploration pressure (prevents all-hold collapse)
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1,
        seed=SEED,
        device="auto",
    )

    total_timesteps = 600_000
    print(f"\nTraining PPO for {total_timesteps} timesteps... saving as {model_path}.zip\n")
    model.learn(total_timesteps=total_timesteps)

    # Save model + normalization stats
    model.save(str(model_path))
    venv.save(str(vecnorm_path))

    print(f"\n✅ Saved model: {model_path}.zip")
    print(f"✅ Saved vecnorm: {vecnorm_path}")

    # Quick sanity: sample actions from the trained model (should not be all zeros)
    obs = venv.reset()
    acts = []
    for _ in range(15):
        a, _ = model.predict(obs, deterministic=True)
        acts.append(int(a))
        obs, _, _, _ = venv.step(a)
    print("Sample actions after training (vec env):", acts)


if __name__ == "__main__":
    main()
