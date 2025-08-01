
import pygame
import numpy as np

# Constants for grid and cell size
CELL_SIZE = 80
GRID_SIZE = 5

# Color definitions for different cell types
COLORS = {
    0: (255, 255, 255),  # Empty
    2: (255, 165, 0),    # Junk Food 🍩
    3: (0, 0, 255),      # Diabetic Student 🧒
    4: (255, 0, 0),      # Anemic Student 🧒
    5: (0, 255, 0),      # Healthy Meal Pack 🍎
    6: (128, 128, 128),  # Missing Ingredients ❌
    7: (255, 255, 0),    # Allergy Alert ⚠️
}

AGENT_COLOR = (0, 200, 0)

# Initialize pygame and font
pygame.init()
FONT_SIZE = int(CELL_SIZE * 0.75)  # Slightly larger for emojis
try:
    FONT = pygame.font.Font("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", FONT_SIZE)
except:
    FONT = pygame.font.SysFont("Segoe UI Emoji", FONT_SIZE)
    print("Warning: Emoji font not found. Falling back to system emoji font.")

def draw_grid(screen, grid, agent_pos, held_meal=None, target=None):
    '''
    Draw the grid with emojis and colored tiles.
    '''
    screen.fill((0, 0, 0))  # Clear screen

    emoji_map = {
        2: "🍩",
        3: "🧒",  # Both types just shown as 🧒
        4: "🧒",
        5: "🍎",
        6: "❌",
        7: "⚠️",
    }

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell_val = grid[i, j]

            # Cell background
            if target is not None and cell_val == target:
                pygame.draw.rect(screen, (255, 0, 255), rect)  # Highlight target
            else:
                pygame.draw.rect(screen, COLORS.get(cell_val, (200, 200, 200)), rect)

            pygame.draw.rect(screen, (50, 50, 50), rect, 2)  # Border

            # Draw emoji if any
            emoji = emoji_map.get(cell_val)
            if emoji:
                emoji_surface = FONT.render(emoji, True, (0, 0, 0))
                emoji_rect = emoji_surface.get_rect(center=rect.center)
                screen.blit(emoji_surface, emoji_rect)

    # Agent box
    agent_rect = pygame.Rect(
        agent_pos[1] * CELL_SIZE + 8,
        agent_pos[0] * CELL_SIZE + 8,
        CELL_SIZE - 16,
        CELL_SIZE - 16
    )
    pygame.draw.rect(screen, AGENT_COLOR, agent_rect)

    # Agent face (based on held meal)
    face = "🙂"
    if held_meal == 1:
        face = "💙"
    elif held_meal == 2:
        face = "❤️"

    agent_surface = FONT.render(face, True, (255, 255, 255))
    agent_rect_center = agent_rect.center
    agent_text_rect = agent_surface.get_rect(center=agent_rect_center)
    screen.blit(agent_surface, agent_text_rect)
