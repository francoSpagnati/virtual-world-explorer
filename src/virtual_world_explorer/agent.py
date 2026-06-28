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
        Rete neurale adibita al controllo continuo.
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


class QLearningAgent: 
    def __init__(self, alpha: float = 0.0005, gamma: float = 0.95, epsilon: float = 1.0) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random(13)
        
        # Aggiornato l'input dimension a 11 e l'output a 2 azioni continue
        self.policy_net = ContinuousPolicyNetwork(input_dim=11, output_dim=2)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state: State) -> np.ndarray:
        """
        Sceglie un'azione continua [v, w] applicando un'esplorazione di tipo rumore gaussiano
        invece del campionamento ad indice casuale (ε-greedy continuo).
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
        Esegue l'aggiornamento dei gradienti sui valori continui predetti.
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32)
        reward_t = torch.tensor([reward], dtype=torch.float32)
        
        # Calcolo dei valori correnti attesi per la combinazione stato-azione continua
        current_pred = self.policy_net(state_t).squeeze(0)
        
        with torch.no_grad():
            if done:
                target = reward_t
            else:
                # Bootstrapping del valore dello stato successivo
                next_pred = self.policy_net(next_state_t).squeeze(0)
                target = reward_t + self.gamma * torch.mean(next_pred)
                
        # Calcolo della perdita basata sulla distanza euclidea dell'output atteso
        loss = self.loss_fn(current_pred, action_t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_exploration(self, minimum: float = 0.02, factor: float = 0.9995) -> None:
        self.epsilon = max(minimum, self.epsilon * factor)