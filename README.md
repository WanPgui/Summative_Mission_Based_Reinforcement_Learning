# Summative_Mission_Based_Reinforcement_Learning
# Reinforcement Learning Agent Comparison: DQN, PPO, A2C

## Link to video: https://drive.google.com/file/d/1ZjwyYKPM3YGkhqN_ASpDZ302sQlbHK1t/view?usp=sharing


## Reinforcement Learning Atari Agent Comparison
Environment:  Gymnasium
Algorithms: Deep Q-Network (DQN), Proximal Policy Optimization (PPO), Advantage Actor-Critic (A2C)
Libraries Used: Stable-Baselines3, Gymnasium, Matplotlib, NumPy, Torch

## Overview
This project trains and evaluates three RL algorithms — **DQN**, **PPO**, and **A2C** — on the `ALE/Breakout-v5` Atari environment using Stable-Baselines3 and Gymnasium. It compares their training stability, rewards, and performance visually and quantitatively.

---

## Setup

1. Install dependencies:
```bash
pip install stable-baselines3[extra] gymnasium[accept-rom-license] ale-py matplotlib numpy
```
2. Imports & Environment Creation
   ```
   from stable_baselines3 import DQN, PPO, A2C
   from stable_baselines3 import DQN, PPO, A2C
   from stable_baselines3.common.vec_env import make_atari_env, VecFrameStack
   from stable_baselines3.common.evaluation import evaluate_policy
   import matplotlib.pyplot as plt
   import numpy as np
3. Training
Trained each model with:
```
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300_000)
model.save("models/")

# Similarly for PPO and A2C with respective hyperparameters.
```
Evaluate every 20k timesteps to log mean reward:
```
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Step {step}: mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
```
4. Rendering
Render a trained model by:
```
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
```
5. Hyperparameters Summary
Algorithm	         Key                       Params
DQN	            Learning Rate: 1e-4,          Gamma: 0.99
PPO	            n_steps: 128,                 Entropy coeff: 0.01
A2C	            n_steps: 5,                   Learning Rate: 7e-4

6. Metrics & Visualization
Episode Reward: Total reward per episode

Mean Reward: Average reward over the last 3 episodes

Best Reward: Highest reward achieved during training

Plot training rewards over timesteps:
```
plt.plot(timesteps, dqn_rewards, label='DQN')
plt.plot(timesteps, ppo_rewards, label='PPO')
plt.plot(timesteps, a2c_rewards, label='A2C')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('Training Reward Progress')
plt.legend()
plt.savefig('reward_progress.png')
plt.show()
```
7. Observations
DQN: Fast and stable learning, reaches peak performance early

PPO: Stable with competitive rewards

A2C: Slower convergence but improves steadily

8. Project structure
   “Summative_Mission_Based_Reinforcement_Learning”.

project_root/

├── environment/

│   ├── custom_env.py            # Custom Gymnasium environment implementation

│   ├── rendering.py             # Visualization GUI components

├── training/

│   ├── dqn_training.py          # Training script for DQN using SB3

│   ├── pg_training.py           # Training script for PPO/other PG using SB3

├── models/

│   ├── dqn/                     # Saved DQN models

│   └── pg/                      # Saved policy gradient models

├── main.py                      # Entry point for running experiments

|__evaluation_gifs/              #Saved gifs 

|   |-- dqn_agent.gif
|   |-- ppo_agent.gif
|   |-- a2c_agent.gif

├── requirements.txt             # Project dependencies

└── README.md                    # Project documentation




