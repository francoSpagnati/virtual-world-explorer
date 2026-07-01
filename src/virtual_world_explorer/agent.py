from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.optim as optim

State = tuple[float, ...]

class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 7, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QLearningAgent: 
    def __init__(self, actions: int = 4, alpha: float = 0.001, gamma: float = 0.9, epsilon: float = 1.0) -> None:
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random(13)
        
        self.policy_net = QNetwork(input_dim=7, output_dim=actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state: State) -> int:
        if self.random.random() < self.epsilon:
            return self.random.randrange(self.actions)
        
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        
        current_q = self.policy_net(state_t)[0][action].unsqueeze(0) 
        reward_t = torch.tensor([reward], dtype=torch.float32)
        
        with torch.no_grad():
            if done:
                target_q = reward_t
            else:
                target_q = reward_t + self.gamma * self.policy_net(next_state_t).max(dim=1)[0]
                
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_exploration(self, minimum: float = 0.05, factor: float = 0.997) -> None:
        self.epsilon = max(minimum, self.epsilon * factor)