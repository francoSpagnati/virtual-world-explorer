from __future__ import annotations

import time
import threading
import queue
import numpy as np
from PIL import Image

from .agent import QLearningAgent
from .env import GridWorldEnv
from .render import OpenGLRenderer
from .owl_vision import OwlVisionDetector

# Impostare a True per usare OWL-ViT in background durante la demo
USE_OWL_VISION = True

owl_frame_queue = queue.Queue(maxsize=1)
owl_result_queue = queue.Queue(maxsize=1)
owl_result = {"dx": 0, "dy": 0, "visible": False}

def owl_worker():
    """Thread in background che esegue l'inferenza OWL-ViT sui frame renderizzati."""
    detector = OwlVisionDetector()
    while True:
        try:
            item = owl_frame_queue.get(timeout=1.0)
            if item is None:
                break
            image, target_name = item
            dx, dy, vis = detector.detect_target(image, target_name)
            owl_result_queue.put({"dx": dx, "dy": dy, "visible": vis})
        except queue.Empty:
            continue


def train_agent(episodes: int = 5000, max_steps: int = 30) -> tuple[GridWorldEnv, QLearningAgent]:
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

    if USE_OWL_VISION:
        print("[OWL] Avvio thread di visione in background...")
        threading.Thread(target=owl_worker, daemon=True).start()

    try:
        state = env.reset()
        recent_positions: list[tuple[int, int]] = []
        step_count = 0
        last_action = None
        while not renderer.should_close():
            # Invio del nuovo frame al thread di visione se abilitato
            if USE_OWL_VISION:
                frame_np = renderer.capture_frame()
                if frame_np is not None:
                    # Svuotiamo code vecchie per sicurezza
                    while not owl_frame_queue.empty():
                        owl_frame_queue.get()
                    while not owl_result_queue.empty():
                        owl_result_queue.get()
                        
                    img = Image.fromarray(frame_np)
                    owl_frame_queue.put((img, env.target_label))
                    
                    # Attendiamo il risultato mantenendo la finestra OpenGL responsiva
                    while owl_result_queue.empty():
                        renderer.poll()
                        if renderer.should_close():
                            break
                        time.sleep(0.01)
                        
                    if renderer.should_close():
                        break
                        
                    res = owl_result_queue.get()
                    owl_result.update(res)

            # Override visivo con i risultati (ora sincroni) di OWL-ViT se abilitato
            current_state = list(state)
            if USE_OWL_VISION:
                current_state[0] = owl_result["dx"]
                current_state[1] = owl_result["dy"]
                current_state[2] = int(owl_result["visible"])
            current_state = tuple(current_state)

            action = _choose_action_without_loop(env, current_state, agent, recent_positions, last_action)
            state, _, done, _ = env.step(action)
            last_action = action
            
            renderer.draw()
            renderer.poll()
            
            # Un piccolo sleep visivo se NON stiamo usando OWL (altrimenti OWL è già abbastanza lento)
            if not USE_OWL_VISION:
                time.sleep(0.35)
            
            # Trova l'oggetto target per stamparne la posizione reale
            target_obj = next((obj for obj in env.objects if obj.label == env.target_label), None)
            target_pos = (target_obj.x, target_obj.y) if target_obj else (None, None)
            
            print(f"[Sim] Agent: ({env.agent_x}, {env.agent_y}) | Target: {target_pos} | OWL dx,dy: ({owl_result['dx']}, {owl_result['dy']}) | Action: {action}")
            
            if done:
                print("Target reached, resetting scene.")
                state = env.reset()
                recent_positions = []
                continue
            recent_positions.append((env.agent_x, env.agent_y))
            if len(recent_positions) > 20:
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


def _choose_action_without_loop(env: GridWorldEnv, state: tuple[int, ...], agent: QLearningAgent, recent_positions: list[tuple[int, int]], last_action: int | None = None) -> int:
    values = agent.q_values[state]
    ranked_actions = sorted(range(agent.actions), key=lambda action: values[action], reverse=True)
    current_position = (env.agent_x, env.agent_y)

    best_action = ranked_actions[0]
    best_score = float("-inf")
    for action in ranked_actions:
        next_position = _preview_position(env, action)
        score = values[action]
        
        # Momentum: solo se NON vede il bersaglio (state[2] == 0)
        # altrimenti rischia di ignorare la sedia se ci passa di fianco
        if state[2] == 0 and last_action is not None and action == last_action:
            score += 0.5
            
        if next_position == current_position:
            score -= 2.0
            
        visit_count = recent_positions.count(next_position)
        if visit_count > 0:
            score -= 3.0 * visit_count

        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def main() -> None:
    env, agent = train_agent()
    run_demo(env, agent)


if __name__ == "__main__":
    main()
