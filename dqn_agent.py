import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Deep Q-Network approximating Q(s, a)."""

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    """Fixed-capacity buffer storing experience tuples for batch sampling."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(
        self,
        state_dim=4,
        action_dim=2,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,       # Calibrated for per-episode decay
        target_update_freq=500,
        batch_size=64,
        buffer_capacity=50000,
        update_every=4,            # Train every 4 environment steps
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_every = update_every

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.total_env_steps = 0
        self.learn_step_counter = 0

    def select_action(self, state, evaluate=False):
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        """Stores experience and triggers training update every N frames."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_env_steps += 1

        if self.total_env_steps % self.update_every == 0:
            if len(self.memory) >= self.batch_size:
                self.learn()

    def learn(self):
        """Samples a batch and updates network weights via Bellman error."""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        curr_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Called once per episode to manage exploration smoothly."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)