import random
import pygame
from pygame.locals import QUIT, KEYDOWN, K_SPACE, K_UP

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_WIDTH, PIPE_GAP,
    PIPE_SPACING, STATE_READY, STATE_PLAYING, STATE_GAME_OVER
)
from entities import Bird, Pipe, Ground


class FlappyGame:

    def __init__(self, screen, clock, assets):
        self.screen = screen
        self.clock = clock
        self.assets = assets

        self.state = STATE_READY
        self.score = 0
        self.bird = None
        self.bird_group = pygame.sprite.GroupSingle()
        self.ground_group = pygame.sprite.Group()
        self.pipe_group = pygame.sprite.Group()

        self.reset()

    def reset(self):
        self.score = 0
        self.state = STATE_PLAYING
        self.bird_group.empty()
        self.ground_group.empty()
        self.pipe_group.empty()

        self.bird = Bird(self.assets)
        self.bird_group.add(self.bird)

        for i in range(2):
            self.ground_group.add(Ground(self.assets, GROUND_WIDTH * i))

        for i in range(2):
            self.spawn_pipe_pair(SCREEN_WIDTH + 300 + (i * PIPE_SPACING))

    def _get_upcoming_pipe(self):
        """Finds the nearest bottom pipe the bird hasn't completely cleared yet."""
        upcoming_pipes = [
            p for p in self.pipe_group
            if not p.inverted and p.rect.right > self.bird.rect.left
        ]
        if upcoming_pipes:
            upcoming_pipes.sort(key=lambda p: p.rect.left)
            return upcoming_pipes[0]
        return None

    def get_observation(self):
        """Constructs and returns the 4-element normalized observation array."""
        target_pipe = self._get_upcoming_pipe()

        if target_pipe is None:
            # Fallback for the rare edge case where no pipe is active yet
            return [0.0, 0.0, 1.0, 0.0]

        # Calculate gap center y
        gap_center_y = target_pipe.rect.top - (PIPE_GAP / 2.0)

        # Compute raw differences
        dx = target_pipe.rect.left - self.bird.rect.right
        dy = gap_center_y - self.bird.rect.centery

        # Normalize features
        norm_bird_y = self.bird.rect.centery / SCREEN_HEIGHT
        norm_velocity = self.bird.velocity / 15.0
        norm_dx = dx / SCREEN_WIDTH
        norm_dy = dy / SCREEN_HEIGHT

        return [norm_bird_y, norm_velocity, norm_dx, norm_dy]

    def spawn_pipe_pair(self, x_pos):
        size = random.randint(100, 300)
        lower_pipe = Pipe(self.assets, False, x_pos, size)
        upper_pipe = Pipe(self.assets, True, x_pos, SCREEN_HEIGHT - size - PIPE_GAP)
        self.pipe_group.add(lower_pipe, upper_pipe)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False

            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if self.state == STATE_READY:
                    self.state = STATE_PLAYING
                    self.bird.flap(self.assets.wing_sound)
                elif self.state == STATE_PLAYING:
                    self.bird.flap(self.assets.wing_sound)
                elif self.state == STATE_GAME_OVER:
                    self.reset()
                    self.state = STATE_READY

        return True

    def update(self):
        if self.state in (STATE_READY, STATE_PLAYING):
            self.ground_group.update()
            first_ground = self.ground_group.sprites()[0]
            if first_ground.is_off_screen:
                self.ground_group.remove(first_ground)
                last_ground = self.ground_group.sprites()[-1]
                self.ground_group.add(Ground(self.assets, last_ground.rect.right))

        if self.state == STATE_READY:
            self.bird.update_animation(speed=6)

        elif self.state == STATE_PLAYING:
            self.bird.update()
            self.pipe_group.update()

            pipes = self.pipe_group.sprites()
            if pipes and pipes[0].is_off_screen:
                self.pipe_group.remove(pipes[0], pipes[1])
                last_x = self.pipe_group.sprites()[-1].rect.x
                self.spawn_pipe_pair(last_x + PIPE_SPACING)

            for pipe in self.pipe_group:
                if not pipe.inverted and not pipe.passed:
                    if self.bird.rect.centerx > pipe.rect.centerx:
                        pipe.passed = True
                        self.score += 1

            ground_hit = pygame.sprite.spritecollide(
                self.bird, self.ground_group, False, pygame.sprite.collide_mask
            )
            pipe_hit = pygame.sprite.spritecollide(
                self.bird, self.pipe_group, False, pygame.sprite.collide_mask
            )
            ceiling_hit = self.bird.rect.top <= 0

            if ground_hit or pipe_hit or ceiling_hit:
                self.assets.hit_sound.play()
                self.state = STATE_GAME_OVER

    def draw(self):
        self.screen.blit(self.assets.background, (0, 0))
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)
        self.bird_group.draw(self.screen)

        if self.state == STATE_READY:
            msg_rect = self.assets.message.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(self.assets.message, msg_rect)

        elif self.state == STATE_PLAYING:
            self.draw_score(self.score, 50)

        elif self.state == STATE_GAME_OVER:
            self.draw_score(self.score, 50)
            self.draw_text("GAME OVER", SCREEN_HEIGHT // 2 - 30, (220, 50, 50))
            self.draw_text("Press SPACE to Restart", SCREEN_HEIGHT // 2 + 20, (255, 255, 255))

        pygame.display.update()

    def draw_score(self, value, y_pos):
        text = str(value)
        center_x = SCREEN_WIDTH // 2
        shadow = self.assets.score_font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, shadow.get_rect(center=(center_x + 2, y_pos + 2)))
        main_surface = self.assets.score_font.render(text, True, (255, 255, 255))
        self.screen.blit(main_surface, main_surface.get_rect(center=(center_x, y_pos)))

    def draw_text(self, text, y_pos, color):
        center_x = SCREEN_WIDTH // 2
        shadow = self.assets.status_font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, shadow.get_rect(center=(center_x + 1, y_pos + 1)))
        main_surface = self.assets.status_font.render(text, True, color)
        self.screen.blit(main_surface, main_surface.get_rect(center=(center_x, y_pos)))