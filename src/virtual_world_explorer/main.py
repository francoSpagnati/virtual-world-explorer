from __future__ import annotations

import time
import threading
import queue
import math
import torch
import numpy as np
from PIL import Image

from .agent import QLearningAgent
from .env import GridWorldEnv
from .render import OpenGLRenderer
from .owl_vision import OwlVisionDetector

USE_OWL_VISION = True

owl_frame_queue = queue.Queue(maxsize=1)
owl_result_queue = queue.Queue(maxsize=1)
owl_result = {"dx": 0.0, "dy": 0.0, "visible": False}

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


def train_agent(episodes: int = 30000, max_steps: int | None = None) -> tuple[GridWorldEnv, QLearningAgent]:
    env = GridWorldEnv()
    agent = QLearningAgent()
    
    if max_steps is None:
        max_steps = int(env.size * env.size)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            # Sceglie un'azione continua [v, w] (array numpy di 2 float)
            action = agent.choose_action(state)
            
            # Passa l'array continuo all'ambiente
            next_state, reward, done, _ = env.step(action)
            
            # Apprende passandogli l'array continuo dell'azione eseguita
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            if done:
                break
                
        # Decadimento dell'esplorazione (rumore gaussiano/epsilon)
        agent.decay_exploration(minimum=0.01, factor=0.999)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1:05d}: reward={episode_reward:.2f} epsilon={agent.epsilon:.2f}")

    return env, agent


def _update_owl_vision_state(env: GridWorldEnv, renderer: OpenGLRenderer, agent_pos: tuple[float, float], episode_owl_cache: dict) -> None:
    if agent_pos in episode_owl_cache:
        owl_result.update(episode_owl_cache[agent_pos])
        return

    frames_list = renderer.capture_frame()
    if frames_list is None:
        return

    while not owl_frame_queue.empty(): owl_frame_queue.get()
    while not owl_result_queue.empty(): owl_result_queue.get()
    
    images = [Image.fromarray(f) for f in frames_list]
    target_names = [env.target_label, "table", "lamp"]
    owl_frame_queue.put((images, target_names))
    
    while owl_result_queue.empty():
        renderer.draw()
        renderer.poll()
        if renderer.should_close(): return
        time.sleep(0.01)
        
    if renderer.should_close(): return
        
    res = owl_result_queue.get()
    owl_result.update(res)
    episode_owl_cache[agent_pos] = res


def run_demo(env: GridWorldEnv, agent: QLearningAgent, steps: int | None = None, max_episodes: int | None = 5) -> None:
    renderer = OpenGLRenderer(env)
    renderer.initialize()
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.0

    if USE_OWL_VISION:
        print("[OWL] Avvio thread di visione in background (8 canali)...")
        threading.Thread(target=owl_worker, daemon=True).start()

    try:
        state = env.reset()
        recent_positions: list[tuple[float, float]] = []
        episode_step_count = 0
        total_step_count = 0
        last_action = None
        episode_owl_cache = {}
        
        with open("movements.log", "w") as log_file:
            log_file.write("=== Simulation Movements Log ===\n")

            while not renderer.should_close():
                agent_pos = (env.agent_x, env.agent_y)
                
                if USE_OWL_VISION:
                    _update_owl_vision_state(env, renderer, agent_pos, episode_owl_cache)
                    if renderer.should_close():
                        break

                    current_state = list(state)
                    if owl_result["visible"]:
                        dx, dy = owl_result["dx"], owl_result["dy"]
                        length = math.hypot(dx, dy)
                        if length > 0:
                            current_state[0] = float(dx) / length
                            current_state[1] = float(dy) / length
                        else:
                            current_state[0] = 0.0
                            current_state[1] = 0.0
                    current_state[2] = int(owl_result["visible"])
                    state = tuple(current_state)

                action = _choose_action_without_loop(env, state, agent, recent_positions, last_action)
                state, _, done, _ = env.step(action)
                last_action = action
                
                renderer.draw()
                renderer.poll()
                
                if not USE_OWL_VISION or agent_pos in episode_owl_cache:
                    time.sleep(0.35)
                
                target_pos = env._target_object().x, env._target_object().y
                log_file.write(f"[Sim] Agent: ({env.agent_x}, {env.agent_y}) | Target: {target_pos} | Action: {action}\n")
                log_file.flush()
                
                episode_step_count += 1
                total_step_count += 1
                
                max_demo_steps = env.size * env.size
                
                if done or episode_step_count >= max_demo_steps:
                    msg = f"Obiettivo raggiunto in {episode_step_count} passi!" if done else f"Limite passi raggiunto ({max_demo_steps})."
                    print(f"{msg} Reset della scena.")
                    
                    if done and max_episodes is not None:
                        max_episodes -= 1
                        if max_episodes <= 0:
                            break
                            
                    state = env.reset()
                    recent_positions.clear()
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


def _preview_position(env: GridWorldEnv, action: int) -> tuple[float, float]:
    x, y = env.agent_x, env.agent_y
    step_size = 0.25
    
    if action == 0:    y -= step_size
    elif action == 1:  y += step_size
    elif action == 2:  x -= step_size
    elif action == 3:  x += step_size
        
    x = max(env.agent_radius, min(env.size - env.agent_radius, x))
    y = max(env.agent_radius, min(env.size - env.agent_radius, y))
    
    for obj in env.objects:
        if obj.label != env.target_label:
            if math.hypot(x - obj.x, y - obj.y) < (env.agent_radius + obj.radius):
                return (env.agent_x, env.agent_y)
    return x, y


def _choose_action_without_loop(env: GridWorldEnv, state: tuple[float, ...], agent: QLearningAgent, recent_positions: list[tuple[float, float]], last_action: np.ndarray | None = None) -> np.ndarray:
    """Versione per spazio delle azioni continuo."""
    # Ottieni l'azione raccomandata dalla rete neurale politica
    action = agent.choose_action(state)
    
    # Se l'agente non vede il target (state[2] == 0) e si muoveva già prima, manteniamo un po' di inerzia
    if state[2] == 0.0 and last_action is not None:
        # Fonde l'azione corrente con l'azione precedente al 70% per dare stabilità rettilinea
        action = 0.3 * action + 0.7 * last_action
        
    # Semplice controllo predittivo: se l'azione lineare porta a una collisione imminente rilevata dal radar
    # costringiamo l'agente a ruotare su se stesso (velocità lineare zero, velocità angolare attiva)
    v_input, w_input = action[0], action[1]
    linear_vel = max(0.0, ((v_input + 1.0) / 2.0) * env.max_linear_velocity)
    
    next_x = env.agent_x + linear_vel * math.cos(env.agent_heading)
    next_y = env.agent_y + linear_vel * math.sin(env.agent_heading)
    
    if env._check_collision_at(next_x, next_y):
        # Ostacolo rilevato! Sterza bruscamente sul posto per evitare il blocco continuo
        action[0] = -1.0 # Forza velocità lineare a zero
        action[1] = 1.0 if w_input >= 0 else -1.0 # Sterza al massimo a destra o sinistra
        
    return action


def main() -> None:
    env, agent = train_agent(episodes=500)
    run_demo(env, agent, max_episodes=6)


if __name__ == "__main__":
    main()