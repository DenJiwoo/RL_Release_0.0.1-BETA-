from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / 'assets'

# Screen Dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Physics & Scroll Speeds (Calibrated for 60 FPS)
GRAVITY = 0.6
JUMP_STRENGTH = 8.0
SCROLL_SPEED = 4

# Entity Dimensions & Spacing
GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100
PIPE_WIDTH = 70
PIPE_HEIGHT = 500
PIPE_GAP = 165
PIPE_SPACING = 220

# Game States
STATE_READY = 0
STATE_PLAYING = 1
STATE_GAME_OVER = 2