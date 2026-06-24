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
            images, target_names = item
            dx, dy, vis = detector.detect_target_multiview(images, target_names)
            owl_result_queue.put({"dx": dx, "dy": dy, "visible": vis})
        except queue.Empty:
            continue


def train_agent(episodes: int = 15000, max_steps: int | None = None) -> tuple[GridWorldEnv, QLearningAgent]:
    
    env = GridWorldEnv()
    agent = QLearningAgent()
    
    if max_steps is None:
        max_steps = env.size * env.size

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
        # Usiamo un decadimento più lento per esplorare meglio lo spazio degli stati
        agent.decay_exploration(minimum=0.01, factor=0.999)
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1:05d}: reward={episode_reward:.2f} epsilon={agent.epsilon:.2f}")

    return env, agent


def run_demo(env: GridWorldEnv, agent: QLearningAgent, steps: int | None = None, max_episodes: int | None = 5) -> None:
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
        episode_step_count = 0
        total_step_count = 0
        last_action = None
        episode_owl_cache = {}
        
        with open("movements.log", "w") as log_file:
            log_file.write("=== Simulation Movements Log ===\n")

            while not renderer.should_close():
                agent_pos = (env.agent_x, env.agent_y)
                
                if USE_OWL_VISION:
                    if agent_pos in episode_owl_cache:
                        owl_result.update(episode_owl_cache[agent_pos])
                    else:
                        # Otteniamo la lista delle 4 inquadrature orizzontali
                        frames_list = renderer.capture_frame()
                        
                        if frames_list is not None:
                            while not owl_frame_queue.empty():
                                owl_frame_queue.get()
                            while not owl_result_queue.empty():
                                owl_result_queue.get()
                            
                            # Invia tutte e 4 le inquadrature per la scansione a 360 gradi
                            images = [Image.fromarray(f) for f in frames_list]
                            target_names = [env.target_label, "table", "lamp"]
                            owl_frame_queue.put((images, target_names))
                            
                            # Ciclo di attesa che continua a renderizzare la vista zenitale principale 
                            # evitando blocchi o schermate nere sull'interfaccia utente
                            while owl_result_queue.empty():
                                renderer.draw()
                                renderer.poll()
                                if renderer.should_close():
                                    break
                                time.sleep(0.01)
                                
                            if renderer.should_close():
                                break
                                
                            res = owl_result_queue.get()
                            owl_result.update(res)
                            episode_owl_cache[agent_pos] = res

                current_state = list(state)
                if USE_OWL_VISION:
                    visible = owl_result["visible"]
                    if visible:
                        current_state[0] = owl_result["dx"]
                        current_state[1] = owl_result["dy"]
                    current_state[2] = int(visible)
                current_state = tuple(current_state)

                action = _choose_action_without_loop(env, current_state, agent, recent_positions, last_action)
                state, _, done, _ = env.step(action)
                last_action = action
                
                # Renderizza la normale vista zenitale d'osservazione
                renderer.draw()
                renderer.poll()
                
                if not USE_OWL_VISION or (USE_OWL_VISION and agent_pos in episode_owl_cache):
                    time.sleep(0.35)
                
                target_obj = next((obj for obj in env.objects if obj.label == env.target_label), None)
                target_pos = (target_obj.x, target_obj.y) if target_obj else (None, None)
                
                log_file.write(f"[Sim] Agent: ({env.agent_x}, {env.agent_y}) | Target: {target_pos} | Action: {action}\n")
                log_file.flush()
                
                episode_step_count += 1
                total_step_count += 1
                
                if done:
                    print(f"Obiettivo raggiunto in {episode_step_count} passi! Reset della scena.")
                    if max_episodes is not None:
                        max_episodes -= 1
                        if max_episodes <= 0:
                            break
                    state = env.reset()
                    recent_positions = []
                    episode_step_count = 0
                    episode_owl_cache.clear()
                    continue
                
                max_demo_steps = env.size * env.size
                if episode_step_count >= max_demo_steps:
                    print(f"Limite passi raggiunto ({max_demo_steps}), l'agente non ha trovato l'obiettivo. Reset della scena.")
                    state = env.reset()
                    recent_positions = []
                    episode_step_count = 0
                    episode_owl_cache.clear()
                    continue

                recent_positions.append((env.agent_x, env.agent_y))
                if len(recent_positions) > 20:
                    recent_positions.pop(0)

                if steps is not None and total_step_count >= steps:
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
    run_demo(env, agent, max_episodes=6)


if __name__ == "__main__":
    main()