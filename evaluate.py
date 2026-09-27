import time
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
import torch

from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent

MODEL_FILE = "best_flappy_model.pth"
NUM_EPISODES = 5

# 1. Initialize human visual mode
env = FlappyBirdEnv(render_mode="human")

# 2. Recreate network structure and load saved weights
agent = DQNAgent(state_dim=4, action_dim=2)

try:
    agent.policy_net.load_state_dict(
        torch.load(MODEL_FILE, map_location=agent.device, weights_only=True)
    )
    print(f"Loaded weights from {MODEL_FILE}")
except FileNotFoundError:
    print(f"Error: Could not find '{MODEL_FILE}'. Run train.py first.")
    env.close()
    raise SystemExit

agent.policy_net.eval()

print("Running evaluation rounds... Close the window or press ESC to exit.")

for episode in range(1, NUM_EPISODES + 1):
    state, info = env.reset()
    episode_reward = 0.0
    running = True

    while running:
        # Check window events to allow clean exit during evaluation
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                env.close()
                raise SystemExit

        # evaluate=True bypasses epsilon and selects argmax Q(s, a)
        action = agent.select_action(state, evaluate=True)

        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            print(f"Round {episode} | Score: {info['score']} | Return: {episode_reward:.1f}")
            time.sleep(1.0)  # Pause briefly on death before restarting
            break

env.close()