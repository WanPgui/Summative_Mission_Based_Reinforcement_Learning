import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys
import argparse

# Add environment to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))
from custom_env import SchoolCanteenEnv

def train_dqn(total_timesteps=200000, save_path='../models/dqn/dqn_model.zip',
              learning_rate=1e-3, buffer_size=10000, batch_size=64, gamma=0.99,
              train_freq=4, target_update_interval=1000, eval_freq=10000, eval_episodes=5):
    train_env = SchoolCanteenEnv()
    eval_env = SchoolCanteenEnv()

    check_env(train_env)  # sanity check

    model = DQN(
        'MlpPolicy', train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        verbose=1
    )

    # Evaluate before training
    try:
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
        print(f"Initial mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    except Exception as e:
        print(f"Warning: evaluation failed before training: {e}")

    timesteps_run = 0
    while timesteps_run < total_timesteps:
        train_chunk = min(eval_freq, total_timesteps - timesteps_run)
        model.learn(total_timesteps=train_chunk, reset_num_timesteps=False)
        timesteps_run += train_chunk

        try:
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
            print(f"After {timesteps_run} timesteps: mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"Warning: evaluation failed during training: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"DQN model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN on SchoolCanteenEnv with hyperparameters')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--buffer', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--train_freq', type=int, default=4, help='Training frequency')
    parser.add_argument('--target_update', type=int, default=1000, help='Target network update interval')
    parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluation frequency during training')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of episodes per evaluation')
    parser.add_argument('--save_path', type=str, default='../models/dqn/dqn_model.zip', help='Path to save the model')

    args = parser.parse_args()
    train_dqn(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        buffer_size=args.buffer,
        batch_size=args.batch,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_path=args.save_path
    )