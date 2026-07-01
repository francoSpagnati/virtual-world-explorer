from __future__ import annotations

from collections import defaultdict
import random


State = tuple[int, ...]


class QLearningAgent:
    def __init__(self, actions: int = 4, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random(13)
        self.q_values = defaultdict(lambda: [0.0 for _ in range(self.actions)])

    def choose_action(self, state: State) -> int:
        if self.random.random() < self.epsilon:
            return self.random.randrange(self.actions)
        values = self.q_values[state]
        return max(range(self.actions), key=lambda action: values[action])

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        current = self.q_values[state][action]
        next_best = 0.0 if done else max(self.q_values[next_state])
        target = reward + self.gamma * next_best
        self.q_values[state][action] = current + self.alpha * (target - current)

    def decay_exploration(self, minimum: float = 0.05, factor: float = 0.997) -> None:
        self.epsilon = max(minimum, self.epsilon * factor)
