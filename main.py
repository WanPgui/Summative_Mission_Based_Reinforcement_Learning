import argparse
import os
import sys

# Add subfolders to sys.path
project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, 'training'))
sys.path.append(os.path.join(project_root, 'environment'))

def main():
    parser = argparse.ArgumentParser(description='Run RL training or evaluation')
    parser.add_argument('--algo', type=str, choices=['dqn', 'ppo', 'a2c', 'random'], default='random',
                        help='Choose the RL algorithm to run: dqn | ppo | a2c | random')
    parser.add_argument('--timesteps', type=int, default=10000, help='Number of training timesteps')
    args = parser.parse_args()

    if args.algo == 'dqn':
        from dqn_training import train_dqn
        train_dqn(total_timesteps=args.timesteps)
    elif args.algo == 'ppo':
        from pg_training import train_ppo
        train_ppo(total_timesteps=args.timesteps)
    elif args.algo == 'a2c':
        from pg_training import train_a2c
        train_a2c(total_timesteps=args.timesteps)
    elif args.algo == 'random':
        from random_agent_demo import random_agent_demo
        random_agent_demo()
    else:
        print('Invalid algorithm choice.')

if __name__ == '__main__':
    main()