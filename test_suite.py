import math
from unittest.mock import patch
from src.virtual_world_explorer.main import train_agent
from src.virtual_world_explorer.env import GridWorldEnv

def run_test_layout(name, positions, max_steps=100):
    print(f"\n--- Running Test Layout: {name} ---")
    
    # Train agent normally (using random positions so it learns general policy)
    print("Training agent...")
    env, agent = train_agent(episodes=15000) # Reduced for faster test suite
    agent.epsilon = 0.0 # Greedy policy
    
    with patch.object(env, '_sample_continuous_positions', return_value=positions):
        state = env.reset()
        
        print(f"Start Agent: ({env.agent_x}, {env.agent_y})")
        target = env._target_object()
        print(f"Target Chair: ({target.x}, {target.y})")
        print("Obstacles:")
        for obj in env.objects:
            if obj.label != "chair":
                print(f"  {obj.label} at ({obj.x}, {obj.y})")
                
        visited = []
        loop_detected = False
        last_action = None
        
        for step in range(max_steps):
            visited.append((env.agent_x, env.agent_y, env.agent_theta))
            
            # Use the actual action selection with momentum and anti-loop
            from src.virtual_world_explorer.main import _choose_action_without_loop
            action = _choose_action_without_loop(env, state, agent, visited[-20:], last_action)
            last_action = action
            
            state, reward, done, _ = env.step(action)
            
            # Check for loops (revisiting same position 3 times)
            current_pos = (env.agent_x, env.agent_y)
            visit_count = sum(1 for p in visited if math.hypot(p[0]-current_pos[0], p[1]-current_pos[1]) < 0.1)
            
            if visit_count >= 3:
                print(f"Step {step}: Loop detected at {current_pos}. Agent is stuck.")
                loop_detected = True
                break
                
            if done:
                print(f"Step {step}: Target reached at {current_pos}!")
                break
                
        if not done and not loop_detected:
            print(f"Failed to reach target within {max_steps} steps. Ended at ({env.agent_x}, {env.agent_y}).")
        
        print(f"Total Steps: {step+1}")
        return done, loop_detected

if __name__ == "__main__":
    # positions format: [agent, chair, table1, lamp1, table2, lamp2, table3, lamp3]
    # Agent and Chair at opposite ends, no obstacles in between
    layout_clear = [
        (1.0, 1.0), # Agent
        (6.0, 6.0), # Chair
        (1.0, 6.0), (1.0, 5.0), (2.0, 6.0), (2.0, 5.0), (6.0, 1.0), (5.0, 1.0) # Obstacles out of the way
    ]
    
    # Agent at bottom left, Chair at top right, Table exactly in the middle blocking direct diagonal path
    layout_blocked = [
        (1.0, 1.0), # Agent
        (5.0, 5.0), # Chair
        (3.0, 3.0), # Table in the middle
        (1.0, 5.0), (2.0, 6.0), (2.0, 5.0), (6.0, 1.0), (5.0, 1.0) # Other obstacles out of the way
    ]
    
    run_test_layout("Clear Path", layout_clear)
    run_test_layout("Blocked Path", layout_blocked)
