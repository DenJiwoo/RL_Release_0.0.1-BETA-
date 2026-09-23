from flappy_env import FlappyBirdEnv

env = FlappyBirdEnv(render_mode="human")
obs, info = env.reset()

print("Initial observation shape:", obs.shape)
print("Initial observation values:", obs)

for step_num in range(100):
    action = env.action_space.sample()  # Random action: 0 or 1
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print(f"Died at step {step_num}, Score: {info['score']}")
        obs, info = env.reset()

env.close()