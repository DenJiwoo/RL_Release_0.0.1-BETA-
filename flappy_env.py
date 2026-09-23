import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, STATE_PLAYING, STATE_GAME_OVER
)
from assets import AssetManager
from game import FlappyGame


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Actions: 0 = Idle, 1 = Flap
        self.action_space = spaces.Discrete(2)

        # Observations: [norm_bird_y, norm_velocity, norm_dx, norm_dy]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.5, -1.0, -1.5], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float32),
            dtype=np.float32
        )

        pygame.init()
        pygame.font.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - Reinforcement Learning")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.assets = AssetManager()
        self.game = FlappyGame(self.screen, self.clock, self.assets)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game.reset()
        obs = np.array(self.game.get_observation(), dtype=np.float32)
        info = {"score": self.game.score}

        return obs, info

    def step(self, action):
        # Keep Pygame responsive to OS events
        pygame.event.pump()

        prev_score = self.game.score

        # 0 = Idle, 1 = Flap
        if action == 1:
            self.game.bird.flap(sound=None)

        self.game.update()

        terminated = (self.game.state == STATE_GAME_OVER)
        truncated = False

        if terminated:
            reward = -10.0
        else:
            reward = 0.1  # Survival incentive
            if self.game.score > prev_score:
                reward += 5.0  # Pipe clearance reward

        obs = np.array(self.game.get_observation(), dtype=np.float32)
        info = {"score": self.game.score}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game.draw()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()