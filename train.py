import collections
import numpy as np
import torch
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent

# Training settings
NUM_EPISODES = 1200
MAX_STEPS_PER_EPISODE = 3000
PRINT_EVERY = 50

env = FlappyBirdEnv(render_mode=None)
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    lr=5e-4,
    gamma=0.99,
    epsilon_start=0.1,
    epsilon_min=0.001,
    epsilon_decay=0.995,     # Smooth drop across ~900 episodes
    target_update_freq=500,
    batch_size=64,
    buffer_capacity=50000,
    update_every=4           # Train every 4 steps
)

best_rolling_avg = 0.0
best_single_score = 0
recent_scores = collections.deque(maxlen=50)

print("Starting training with episode-level decay...")
print(f"{'Episode':<10} | {'Score':<8} | {'Avg (Last 50)':<15} | {'Best Avg':<10} | {'Epsilon':<8}")
print("-" * 65)

for episode in range(1, NUM_EPISODES + 1):
    state, info = env.reset()

    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.step(state, action, reward, next_state, terminated)

        state = next_state
        if terminated or truncated:
            break

    score = info["score"]
    recent_scores.append(score)
    current_avg = float(np.mean(recent_scores))

    if score > best_single_score:
        best_single_score = score

    # Save weights whenever 50-episode baseline hits a new peak
    if len(recent_scores) >= 50 and current_avg > best_rolling_avg:
        best_rolling_avg = current_avg
        torch.save(agent.policy_net.state_dict(), "best_flappy_model.pth")

    # Decay exploration once per episode
    agent.decay_epsilon()

    if episode % PRINT_EVERY == 0:
        print(
            f"{episode:<10} | {score:<8} | {current_avg:<15.2f} | "
            f"{best_rolling_avg:<10.2f} | {agent.epsilon:<8.3f}"
        )

print(f"\nTraining finished!")
print(f"All-time Peak Score: {best_single_score}")
print(f"Best Rolling 50-Episode Average: {best_rolling_avg:.2f}")

env.close()