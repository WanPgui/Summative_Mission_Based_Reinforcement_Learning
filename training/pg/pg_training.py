pg_training_code = """
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys
import argparse

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# Add environment to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))
from custom_env import SchoolCanteenEnv

def make_env():
    env = SchoolCanteenEnv()
    return gym.wrappers.RecordEpisodeStatistics(env)

def train_model(algorithm, policy, total_timesteps, save_path, eval_freq,
                eval_episodes, verbose=1, **kwargs):

    train_env = make_env()
    eval_env = make_env()
    check_env(train_env.unwrapped)

    model = algorithm(policy, train_env, verbose=verbose, **kwargs)

    print(f"Initial evaluation for {algorithm.__name__}...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
    print(f"[{algorithm.__name__}] Initial mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    best_mean_reward = mean_reward
    timesteps_run = 0

    while timesteps_run < total_timesteps:
        chunk = min(eval_freq, total_timesteps - timesteps_run)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        timesteps_run += chunk

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
        print(f"[{algorithm.__name__}] Step {timesteps_run}: mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            print(f" [{algorithm.__name__}] New best model saved with reward {best_mean_reward:.2f}")

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[{algorithm.__name__}] Final model saved.")

    print(f"[{algorithm.__name__}] Training complete. Total timesteps: {timesteps_run}, Best reward: {best_mean_reward:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO and A2C on SchoolCanteenEnv with hyperparameters')

    # Common arguments
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--eval_freq', type=int, default=20_000)
    parser.add_argument('--eval_episodes', type=int, default=5)

    # PPO-specific arguments
    parser.add_argument('--ppo_lr', type=float, default=3e-4)
    parser.add_argument('--ppo_gamma', type=float, default=0.99)
    parser.add_argument('--ppo_batch', type=int, default=64)
    parser.add_argument('--ppo_nsteps', type=int, default=2048)
    parser.add_argument('--ppo_entcoef', type=float, default=0.01)
    parser.add_argument('--ppo_save_path', type=str, default='../models/pg/ppo_model.zip')

    # A2C-specific arguments
    parser.add_argument('--a2c_lr', type=float, default=7e-4)
    parser.add_argument('--a2c_gamma', type=float, default=0.99)
    parser.add_argument('--a2c_save_path', type=str, default='../models/pg/a2c_model.zip')

    args = parser.parse_args()

    # Train PPO
    train_model(
        PPO, 'MlpPolicy',
        total_timesteps=args.timesteps,
        save_path=args.ppo_save_path,
        learning_rate=args.ppo_lr,
        gamma=args.ppo_gamma,
        batch_size=args.ppo_batch,
        n_steps=args.ppo_nsteps,
        ent_coef=args.ppo_entcoef,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes
    )

    # Train A2C
    train_model(
        A2C, 'MlpPolicy',
        total_timesteps=args.timesteps,
        save_path=args.a2c_save_path,
        learning_rate=args.a2c_lr,
        gamma=args.a2c_gamma,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes
    )

"""
with open("training/pg_training.py", "w") as f:
    f.write(pg_training_code.strip())

print("pg_training.py updated with hyperparameter tuning and evaluation.")
