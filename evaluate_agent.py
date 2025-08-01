
import os
import sys
import imageio
import numpy as np
import pygame

# Use dummy video driver for pygame to avoid display errors in headless envs
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Add environment and models path
sys.path.append(os.path.join(os.getcwd(), 'environment'))

from custom_env import SchoolCanteenEnv
from rendering import draw_grid, CELL_SIZE, GRID_SIZE  # Import draw_grid and constants

# Emoji strings matching rendering.py
EMOJI_MAP = {
    2: "🍩",
    3: "🧒",
    4: "🧒",
    5: "🍎",
    6: "❌",
    7: "⚠️",
}

AGENT_FACES = {
    0: "🙂",   # None held meal
    1: "💙",   # Diabetic meal
    2: "❤️",   # Anemic meal
}

# Initialize pygame font with emoji support — scaled to CELL_SIZE
pygame.init()
FONT_SIZE = int(CELL_SIZE * 0.65)
try:
    FONT = pygame.font.Font("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", FONT_SIZE)
except Exception:
    FONT = pygame.font.SysFont("arial", FONT_SIZE)
    print("Warning: Emoji font not found. Falling back to Arial.")

def overlay_emojis(frame_np, grid, agent_pos, held_meal):
    """
    Overlay emojis onto the frame numpy array using pygame font rendering.
    """
    try:
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)

        # Convert numpy array to pygame surface
        surface = pygame.image.frombuffer(frame_np.tobytes(), frame_np.shape[1::-1], 'RGB').convert_alpha()

        # Draw emojis on each grid cell
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell_val = grid[i, j]
                emoji = EMOJI_MAP.get(cell_val)
                if emoji:
                    text_surf = FONT.render(emoji, True, (0, 0, 0))
                    x_pos = j * CELL_SIZE + (CELL_SIZE - text_surf.get_width()) // 2
                    y_pos = i * CELL_SIZE + (CELL_SIZE - text_surf.get_height()) // 2
                    surface.blit(text_surf, (x_pos, y_pos))

        # Draw agent face emoji
        face_emoji = AGENT_FACES.get(held_meal if held_meal is not None else 0, "🙂")
        face_surf = FONT.render(face_emoji, True, (255, 255, 255))
        x_agent = agent_pos[1] * CELL_SIZE + (CELL_SIZE - face_surf.get_width()) // 2
        y_agent = agent_pos[0] * CELL_SIZE + (CELL_SIZE - face_surf.get_height()) // 2
        surface.blit(face_surf, (x_agent, y_agent))

        # Convert back to numpy array (transpose to (height, width, channels))
        data = pygame.surfarray.array3d(surface)
        data = np.transpose(data, (1, 0, 2))
        return data

    except Exception as e:
        print(f"Emoji overlay failed: {e}")
        return frame_np

def evaluate_model(model_path, algo_name, num_episodes=3, gif_path=None):
    if algo_name.lower() == 'dqn':
        from stable_baselines3 import DQN as ModelClass
    elif algo_name.lower() == 'ppo':
        from stable_baselines3 import PPO as ModelClass
    elif algo_name.lower() == 'a2c':
        from stable_baselines3 import A2C as ModelClass
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    if gif_path is None:
        raise ValueError("Please specify gif_path to save the performance GIF.")

    env = SchoolCanteenEnv()
    model = ModelClass.load(model_path)

    screen = pygame.display.set_mode((CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE))
    clock = pygame.time.Clock()

    frames = []
    episode_rewards = []
    episode_steps = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            agent_pos = obs[:2]
            grid_flat = obs[2:-1]
            grid = grid_flat.reshape((GRID_SIZE, GRID_SIZE))
            held_meal = obs[-1]

            draw_grid(screen, grid, agent_pos, held_meal)
            pygame.display.flip()

            data = pygame.surfarray.array3d(screen)
            data = np.transpose(data, (1, 0, 2))

            # Overlay emojis
            data = overlay_emojis(data, grid, agent_pos, held_meal)

            frames.append(data)

            clock.tick(env.metadata.get("render_fps", 10))

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        print(f"Episode {ep+1}: Total Reward = {total_reward}, Steps = {steps}")

    pygame.quit()

    imageio.mimsave(gif_path, frames, fps=env.metadata.get("render_fps", 10))
    print(f"Saved performance GIF: {gif_path}")

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = sum(episode_steps) / len(episode_steps)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Average Steps over {num_episodes} episodes: {avg_steps:.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate RL agent and save GIF')
    parser.add_argument('--model', type=str, required=True, help='Path to saved RL model (.zip)')
    parser.add_argument('--algo', type=str, choices=['dqn', 'ppo', 'a2c'], required=True, help='Algorithm type')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--gif', type=str, required=True, help='Output GIF filename (e.g. dqn_agent.gif)')
    args = parser.parse_args()
    evaluate_model(args.model, args.algo, args.episodes, args.gif)
