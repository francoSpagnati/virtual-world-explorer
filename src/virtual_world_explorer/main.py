from __future__ import annotations

import time

from .agent import QLearningAgent
from .env import GridWorldEnv
from .render import OpenGLRenderer


def train_agent(episodes: int = 5000, max_steps: int = 50) -> tuple[GridWorldEnv, QLearningAgent]:
    env = GridWorldEnv()
    agent = QLearningAgent()

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        agent.decay_exploration()
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1:03d}: reward={episode_reward:.2f} epsilon={agent.epsilon:.2f}")

    return env, agent


def run_demo(env: GridWorldEnv, agent: QLearningAgent, steps: int | None = None) -> None:
    renderer = OpenGLRenderer(env)
    renderer.initialize()
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        state = env.reset()
        recent_positions: list[tuple[int, int]] = []
        step_count = 0
        while not renderer.should_close():
            action = _choose_action_without_loop(env, state, agent, recent_positions)
            state, _, done, _ = env.step(action)
            renderer.draw()
            renderer.poll()
            time.sleep(0.35)
            if done:
                print("Target reached, resetting scene.")
                state = env.reset()
                recent_positions = []
                continue
            recent_positions.append((state[0], state[1]))
            if len(recent_positions) > 4:
                recent_positions.pop(0)
            step_count += 1
            if steps is not None and step_count >= steps:
                break
    finally:
        agent.epsilon = previous_epsilon
        renderer.shutdown()


def _preview_position(env: GridWorldEnv, action: int) -> tuple[int, int]:
    x, y = env.agent_x, env.agent_y
    if action == 0:
        y = max(0, y - 1)
    elif action == 1:
        y = min(env.size - 1, y + 1)
    elif action == 2:
        x = max(0, x - 1)
    elif action == 3:
        x = min(env.size - 1, x + 1)
    if any(
        obj.label != env.target_label and obj.x == x and obj.y == y
        for obj in env.objects
    ):
        return (env.agent_x, env.agent_y)
    return x, y


def _choose_action_without_loop(env: GridWorldEnv, state: tuple[int, ...], agent: QLearningAgent, recent_positions: list[tuple[int, int]]) -> int:
    values = agent.q_values[state]
    ranked_actions = sorted(range(agent.actions), key=lambda action: values[action], reverse=True)
    current_position = (env.agent_x, env.agent_y)
    recent_lookup = set(recent_positions[-4:])
    last_position = recent_positions[-1] if recent_positions else None

    best_action = ranked_actions[0]
    best_score = float("-inf")
    for action in ranked_actions:
        next_position = _preview_position(env, action)
        score = values[action]
        if next_position == current_position:
            score -= 1.0
        if next_position in recent_lookup:
            score -= 0.7
        if last_position is not None and next_position == last_position:
            score -= 0.5
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def main() -> None:
    env, agent = train_agent()
    run_demo(env, agent)


if __name__ == "__main__":
    main()
