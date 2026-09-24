import os
import numpy as np
import torch
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent

# Benchmark Settings
NUM_EPISODES = 20
SCORE_CAP = 100
MODEL_PATH = "best_flappy_model.pth"

# Suppress SDL audio in headless mode
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Initialize headless environment
env = FlappyBirdEnv(render_mode=None)

# Initialize agent structure with pure greedy evaluation settings
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=0.0,
    epsilon_min=0.0,
    epsilon_decay=1.0,
)

# Load checkpoint
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Checkpoint file '{MODEL_PATH}' not found. "
        "Run train.py first to generate the progressive model checkpoint."
    )

checkpoint = torch.load(MODEL_PATH, map_location=agent.device)
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    agent.policy_net.load_state_dict(checkpoint["model_state"])
else:
    agent.policy_net.load_state_dict(checkpoint)

agent.policy_net.eval()

# Benchmark tracking
scores = []
steps_list = []
max_speeds_list = []
cap_reached_count = 0

print(f"Starting Progressive Benchmark: {NUM_EPISODES} deterministic rounds (Epsilon = 0.0, Cap = {SCORE_CAP})")
print(f"{'Round':<8} | {'Score':<8} | {'Max Speed':<12} | {'Steps':<8} | {'Result':<18}")
print("-" * 62)

for episode in range(1, NUM_EPISODES + 1):
    state, info = env.reset(seed=episode)
    ep_score = 0
    ep_steps = 0
    ep_max_speed = info.get("pipe_speed", 4.0)

    while True:
        # Pure greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        ep_score = info.get("score", 0)
        current_speed = info.get("pipe_speed", 4.0)
        if current_speed > ep_max_speed:
            ep_max_speed = current_speed

        ep_steps += 1
        state = next_state

        if ep_score >= SCORE_CAP:
            result_str = f"Capped ({SCORE_CAP} pts)"
            cap_reached_count += 1
            break
        elif terminated or truncated:
            result_str = f"Crashed at {ep_score}"
            break

    scores.append(ep_score)
    steps_list.append(ep_steps)
    max_speeds_list.append(ep_max_speed)
    print(
        f"{episode:<8} | {ep_score:<8} | "
        f"{f'{ep_max_speed:.1f} px/f':<12} | "
        f"{ep_steps:<8} | {result_str:<18}"
    )

env.close()

# Statistical Calculations
mean_score = float(np.mean(scores))
median_score = float(np.median(scores))
std_score = float(np.std(scores))
min_score = int(np.min(scores))
max_score = int(np.max(scores))
success_rate = (cap_reached_count / NUM_EPISODES) * 100.0
avg_steps = float(np.mean(steps_list))
avg_max_speed = float(np.mean(max_speeds_list))
peak_speed_achieved = float(np.max(max_speeds_list))

print("\n" + "=" * 50)
print("PROGRESSIVE BENCHMARK SUMMARY STATISTICS")
print("=" * 50)
print(f"Episodes Evaluated    : {NUM_EPISODES}")
print(f"Target Score Cap       : {SCORE_CAP}")
print(f"Success Rate           : {success_rate:.2f}% ({cap_reached_count}/{NUM_EPISODES} reached cap)")
print(f"Mean Score             : {mean_score:.2f}")
print(f"Median Score           : {median_score:.2f}")
print(f"Standard Deviation     : {std_score:.2f}")
print(f"Min / Max Score        : {min_score} / {max_score}")
print(f"Avg Steps Survived     : {avg_steps:.2f}")
print(f"Avg Max Speed Reached  : {avg_max_speed:.2f} px/frame")
print(f"Peak Speed Tested      : {peak_speed_achieved:.1f} px/frame")
print("=" * 50)