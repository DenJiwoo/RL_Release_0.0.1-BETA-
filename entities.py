import pygame
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, GRAVITY,
    JUMP_STRENGTH, SCROLL_SPEED, GROUND_HEIGHT
)


class Bird(pygame.sprite.Sprite):

    def __init__(self, assets):
        super().__init__()
        self.frames = assets.bird_frames
        self.frame_idx = 0
        self.image = self.frames[self.frame_idx]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2))

        self.velocity = 0.0
        self.anim_timer = 0

    def flap(self, sound=None):
        self.velocity = -JUMP_STRENGTH
        if sound:
            sound.play()

    def update_animation(self, speed=5):
        self.anim_timer += 1
        if self.anim_timer % speed == 0:
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            self.image = self.frames[self.frame_idx]
            self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.update_animation()
        self.velocity += GRAVITY
        self.rect.y += int(self.velocity)


class Pipe(pygame.sprite.Sprite):

    def __init__(self, assets, inverted, x_pos, y_size):
        super().__init__()
        self.inverted = inverted
        self.passed = False
        self.speed = 4.0

        if inverted:
            self.image = pygame.transform.flip(assets.pipe, False, True)
            self.rect = self.image.get_rect()
            self.rect.x = x_pos
            self.rect.y = -(self.rect.height - y_size)
        else:
            self.image = assets.pipe
            self.rect = self.image.get_rect()
            self.rect.x = x_pos
            self.rect.y = SCREEN_HEIGHT - y_size

        self.mask = pygame.mask.from_surface(self.image)
        # Float position tracker for accurate sub-pixel movement
        self.x = float(x_pos)

    def update(self):
        self.x -= getattr(self, "speed", 4.0)
        self.rect.x = int(round(self.x))

    @property
    def is_off_screen(self):
        return self.rect.right < 0


class Ground(pygame.sprite.Sprite):

    def __init__(self, assets, x_pos):
        super().__init__()
        self.image = assets.ground
        self.rect = self.image.get_rect(topleft=(x_pos, SCREEN_HEIGHT - GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.speed = 4.0
        # Float position tracker to prevent ground seam gaps
        self.x = float(x_pos)

    def update(self):
        self.x -= getattr(self, "speed", 4.0)
        self.rect.x = int(round(self.x))

    @property
    def is_off_screen(self):
        return self.rect.right < 0