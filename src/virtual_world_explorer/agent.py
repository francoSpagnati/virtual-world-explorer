from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

State = tuple[float, ...]

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 11, output_dim: int = 2):
        """
        Rete neurale adibita al controllo continuo (Actor).
        Input_dim = 11 (esteso per il radar a 8 direzioni)
        Output_dim = 2 (Linear velocity, Angular velocity)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh() # Forza l'output continuo nell'intervallo [-1.0, 1.0]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int = 11, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], 1))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class QLearningAgent: 
    def __init__(self, alpha: float = 0.0005, gamma: float = 0.95, epsilon: float = 1.0) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random(13)
        
        self.policy_net = ContinuousPolicyNetwork(input_dim=11, output_dim=2)
        self.target_policy_net = ContinuousPolicyNetwork(input_dim=11, output_dim=2)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        
        self.critic_net = CriticNetwork(input_dim=11, action_dim=2)
        self.target_critic_net = CriticNetwork(input_dim=11, action_dim=2)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=alpha)
        
        self.memory = ReplayBuffer(capacity=50000)
        self.batch_size = 64
        self.tau = 0.005

    def choose_action(self, state: State) -> np.ndarray:
        """
        Sceglie un'azione continua [v, w] applicando un'esplorazione di tipo rumore gaussiano.
        """
        if self.random.random() < self.epsilon:
            # Esplorazione: genera forze di controllo casuali tra -1 e 1
            return np.array([self.random.uniform(-1.0, 1.0), self.random.uniform(-1.0, 1.0)], dtype=np.float32)
        
        # Sfruttamento della rete neurale
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_values = self.policy_net(state_t).squeeze(0).cpu().numpy()
            return action_values

    def learn(self, state: State, action: np.ndarray | list[float], reward: float, next_state: State, done: bool) -> None:
        """
        Esegue l'aggiornamento dei gradienti usando DDPG (Deep Deterministic Policy Gradient).
        """
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        state_t = torch.FloatTensor(states)
        action_t = torch.FloatTensor(actions)
        reward_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_state_t = torch.FloatTensor(next_states)
        done_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            next_action_t = self.target_policy_net(next_state_t)
            target_q = self.target_critic_net(next_state_t, next_action_t)
            target_q = reward_t + (1 - done_t) * self.gamma * target_q
            
        current_q = self.critic_net(state_t, action_t)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic_net(state_t, self.policy_net(state_t)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def decay_exploration(self, minimum: float = 0.02, factor: float = 0.9995) -> None:
        self.epsilon = max(minimum, self.epsilon * factor)