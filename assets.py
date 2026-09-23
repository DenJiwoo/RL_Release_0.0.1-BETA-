import pygame
from config import (
    ASSETS_DIR, SCREEN_WIDTH, SCREEN_HEIGHT,
    PIPE_WIDTH, PIPE_HEIGHT, GROUND_WIDTH, GROUND_HEIGHT
)


def load_image(path, with_alpha=False):
    """Loads an image and converts it only if an active display surface exists."""
    surface = pygame.image.load(path)
    if pygame.display.get_surface() is not None:
        return surface.convert_alpha() if with_alpha else surface.convert()
    return surface


class AssetManager:

    def __init__(self):
        sprites_dir = ASSETS_DIR / 'sprites'
        audio_dir = ASSETS_DIR / 'audio'

        # Background & UI
        self.background = load_image(sprites_dir / 'background-day.png')
        self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.message = load_image(sprites_dir / 'message.png', with_alpha=True)

        # Obstacles & Environment
        self.pipe = load_image(sprites_dir / 'pipe-green.png', with_alpha=True)
        self.pipe = pygame.transform.scale(self.pipe, (PIPE_WIDTH, PIPE_HEIGHT))

        self.ground = load_image(sprites_dir / 'base.png', with_alpha=True)
        self.ground = pygame.transform.scale(self.ground, (GROUND_WIDTH, GROUND_HEIGHT))

        # Bird Frames
        self.bird_frames = [
            load_image(sprites_dir / 'bluebird-upflap.png', with_alpha=True),
            load_image(sprites_dir / 'bluebird-midflap.png', with_alpha=True),
            load_image(sprites_dir / 'bluebird-downflap.png', with_alpha=True)
        ]

        # Audio Effects
        self.wing_sound = pygame.mixer.Sound(audio_dir / 'wing.wav')
        self.hit_sound = pygame.mixer.Sound(audio_dir / 'hit.wav')

        # Fonts
        self.score_font = pygame.font.Font(None, 52)
        self.status_font = pygame.font.Font(None, 36)