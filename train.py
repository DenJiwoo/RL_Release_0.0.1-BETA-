import collections
import os
import numpy as np
import torch

# Suppress audio driver
os.environ["SDL_AUDIODRIVER"] = "dummy"

from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent

# Model paths
LOAD_MODEL_PATH = "best_flappy_model.pth"
SAVE_MODEL_PATH = "progressive_flappy_model.pth"

# Training Settings
NUM_EPISODES = 1500
MAX_STEPS_PER_EPISODE = 3000
PRINT_EVERY = 50
WARMUP_STEPS = 2500

# Minimum rolling average threshold before allowing disk saves
MIN_SAVE_THRESHOLD = 8.0

# Initialize headless environment
env = FlappyBirdEnv(render_mode=None)

# Agent configuration calibrated for stable fine-tuning
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    lr=3e-5,
    gamma=0.99,
    epsilon_start=0.05,
    epsilon_min=0.01,
    epsilon_decay=0.997,
    target_update_freq=500,
    batch_size=64,
    buffer_capacity=50000,
    update_every=4,
)

# Load baseline weights as a warm start
if os.path.exists(LOAD_MODEL_PATH):
    print(f"Loading baseline weights from: {LOAD_MODEL_PATH}")
    checkpoint = torch.load(LOAD_MODEL_PATH, map_location=agent.device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        agent.policy_net.load_state_dict(checkpoint["model_state"])
        agent.target_net.load_state_dict(checkpoint["model_state"])
    else:
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(checkpoint)
    print("Baseline weights loaded successfully.")
else:
    raise FileNotFoundError(f"Required baseline model '{LOAD_MODEL_PATH}' was not found.")

# Warm up replay buffer using pure greedy baseline actions
if hasattr(agent, "memory") and len(agent.memory) < WARMUP_STEPS:
    print(f"Warming up replay buffer ({WARMUP_STEPS} steps using greedy baseline)...")
    original_eps = agent.epsilon
    agent.epsilon = 0.0

    warmup_state, _ = env.reset()
    for _ in range(WARMUP_STEPS):
        warmup_action = agent.select_action(warmup_state)
        next_s, rew, term, trunc, _ = env.step(warmup_action)

        if hasattr(agent.memory, "push"):
            agent.memory.push(warmup_state, warmup_action, rew, next_s, term)
        elif hasattr(agent.memory, "append"):
            agent.memory.append((warmup_state, warmup_action, rew, next_s, term))

        if term or trunc:
            warmup_state, _ = env.reset()
        else:
            warmup_state = next_s

    agent.epsilon = original_eps
    print(f"Buffer warmup complete. Epsilon set to {agent.epsilon:.3f}. Starting training.")


def verify_greedy_policy(agent_obj, env_obj):
    """
    Verifies that the policy at epsilon=0 can clear at least 1 pipe
    without plunging into the ground at spawn coordinates.
    """
    val_state, _ = env_obj.reset(seed=42)
    for _ in range(300):
        with torch.no_grad():
            s_t = torch.FloatTensor(val_state).unsqueeze(0).to(agent_obj.device)
            act = agent_obj.policy_net(s_t).argmax(dim=1).item()
        next_s, _, term, trunc, val_info = env_obj.step(act)
        if term or trunc:
            return False
        if val_info.get("score", 0) >= 1:
            return True
        val_state = next_s
    return False


# Main Training Loop
best_rolling_avg = MIN_SAVE_THRESHOLD
best_single_score = 0
recent_scores = collections.deque(maxlen=50)
model_saved = False

print("\nStarting stabilized progressive difficulty training...")
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

    # Guarded model saving
    if len(recent_scores) >= 50 and current_avg > best_rolling_avg:
        if verify_greedy_policy(agent, env):
            best_rolling_avg = current_avg
            save_payload = {
                "model_state": agent.policy_net.state_dict(),
                "best_rolling_avg": best_rolling_avg,
                "peak_score": best_single_score,
                "episodes_trained": episode,
            }
            torch.save(save_payload, SAVE_MODEL_PATH)
            model_saved = True
            print(f"--> [SAVED] Ep {episode:04d} | New Best Avg: {current_avg:.2f} | Peak: {best_single_score}")

    agent.decay_epsilon()

    if episode % PRINT_EVERY == 0:
        print(
            f"{episode:<10} | {score:<8} | {current_avg:<15.2f} | "
            f"{best_rolling_avg:<10.2f} | {agent.epsilon:<8.3f}"
        )

print("\nTraining complete.")
print(f"Progressive Peak Score: {best_single_score}")
print(f"Final Best Rolling Average: {best_rolling_avg:.2f}")

if model_saved:
    print(f"Successfully saved verified checkpoint to: {SAVE_MODEL_PATH}")
else:
    print(f"Warning: No model beat the {MIN_SAVE_THRESHOLD:.1f} threshold with a passing greedy check.")

env.close()