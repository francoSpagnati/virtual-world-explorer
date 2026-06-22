from virtual_world_explorer.main import train_agent, _choose_action_without_loop
from virtual_world_explorer.env import GridWorldEnv

def test_demo():
    print("Training agent...")
    env, agent = train_agent(episodes=5000)
    print("Training finished. Running demo...")
    
    agent.epsilon = 0.0
    state = env.reset()
    recent_positions = []
    
    for step in range(50):
        action = _choose_action_without_loop(env, state, agent, recent_positions)
        print(f"Step {step}: Agent at ({env.agent_x}, {env.agent_y}), Target at {env._target_object().x, env._target_object().y}")
        print(f"         State: {state}")
        print(f"         Q-values: {[f'{v:.3f}' for v in agent.q_values[state]]}")
        print(f"         Action chosen: {action}")
        state, reward, done, info = env.step(action)
        print(f"         Reward: {reward:.3f}, Done: {done}")
        
        if done:
            print("Target reached!")
            break
            
        recent_positions.append((env.agent_x, env.agent_y))
        if len(recent_positions) > 4:
            recent_positions.pop(0)

if __name__ == "__main__":
    test_demo()
