# tests/test_env_random.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.data_loader import generate_synthetic_btc_minute_data, split_into_days
from env.kalshi_env import KalshiBTCHourlyEnv


def main():
    # 1. Generate synthetic BTC data for a few days
    df = generate_synthetic_btc_minute_data(num_days=3)
    day_list = split_into_days(df)

    # 2. Create environment
    env = KalshiBTCHourlyEnv(day_data_list=day_list)

    # 3. Run one random episode
    obs, info = env.reset()
    total_reward = 0.0

    while True:
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print("Episode finished.")
            print("Total reward:", total_reward)
            print("Final info:", info)
            break


if __name__ == "__main__":
    main()
