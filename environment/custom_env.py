import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SchoolCanteenEnv(gym.Env):
    \"""
    Custom Gym environment for School Canteen Nutrition Allocation.

    Grid Legend:
    0: Empty
    1: Wall / Impassable (outside grid)
    2: Junk Food 🍩
    3: Diabetic Student 🧒D (Goal 1)
    4: Anemic Student 🧒A (Goal 2)
    5: Healthy Meal Pack 🍎
    6: Missing Ingredients ❌
    7: Allergy Alert Station ⚠️
    \"""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(SchoolCanteenEnv, self).__init__()

        self.grid_size = 5
        self.action_space = spaces.Discrete(6)  # up, down, left, right, pick-up, deliver

        # Observation: agent position + flattened grid + current held meal (0:none,1:diabetic,2:anemic)
        obs_low = np.array([0, 0] + [0] * (self.grid_size ** 2) + [0])
        obs_high = np.array([self.grid_size - 1, self.grid_size - 1] + [7] * (self.grid_size ** 2) + [2])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)

        self.render_mode = render_mode
        self.window = None

        # Initialize the fixed grid layout
        self._build_grid()

        self.agent_pos = [0, 0]
        self.held_meal = 0  # 0: none, 1: diabetic meal, 2: anemic meal
        self.target_student = None

        self.step_count = 0
        self.max_steps = 100  # Increased max steps for better learning

    def _build_grid(self):
        # Static grid layout
        self.grid = np.array([
            [0,   2,   0,   6,   0],
            [3,   0,   7,   0,   0],
            [0,   0,   0,   2,   5],
            [0,   6,   0,   0,   0],
            [0,   0,   4,   0,   0]
        ], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.held_meal = 0

        # Randomly choose target student (diabetic or anemic)
        self.target_student = random.choice([3, 4])
        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.array(self.agent_pos + self.grid.flatten().tolist() + [self.held_meal], dtype=np.int32)
        return obs

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False
        info = {}

        x, y = self.agent_pos

        if action in [0, 1, 2, 3]:  # Move
            new_x, new_y = x, y
            if action == 0:  # Up
                new_x = max(0, x - 1)
            elif action == 1:  # Down
                new_x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # Left
                new_y = max(0, y - 1)
            elif action == 3:  # Right
                new_y = min(self.grid_size - 1, y + 1)

            tile = self.grid[new_x, new_y]
            self.agent_pos = [new_x, new_y]

            # Moderate shaping rewards for visiting important tiles
            if tile == 5:  # Healthy Meal Pack
                reward += 1
            elif tile in [3, 4]:  # Target Students
                reward += 0.5

            # Softer negative shaping for bad tiles
            if tile == 2:  # Junk Food
                reward -= 2
            elif tile == 7:  # Allergy Alert
                reward -= 3
            elif tile == 6:  # Missing ingredients
                reward -= 2

        elif action == 4:  # Pick up meal
            current_tile = self.grid[x, y]
            if current_tile == 5:  # Healthy Meal Pack
                if self.target_student == 3:
                    self.held_meal = 1  # Diabetic meal
                elif self.target_student == 4:
                    self.held_meal = 2  # Anemic meal
                reward += 5
            elif current_tile == 6:
                reward -= 2
            else:
                reward -= 1

        elif action == 5:  # Deliver meal
            current_tile = self.grid[x, y]
            if current_tile == self.target_student:
                if self.held_meal == 0:
                    reward -= 3
                elif (self.held_meal == 1 and self.target_student == 3) or \
                     (self.held_meal == 2 and self.target_student == 4):
                    reward += 20  # Big success reward
                    done = True
                else:
                    reward -= 5
                    done = True
            else:
                reward -= 1

        else:
            reward -= 1

        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, info

    def render(self):
        # placeholder; see rendering.py
        pass

    def close(self):
        if self.window:
            import pygame
            pygame.quit()
            self.window = None
