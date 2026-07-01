# Semantic Room Explorer - continuous_mov

This branch evaluates a continuous unicycle model (v, w) for agent control. The expanded state space caused severe policy degeneration (spinning local minima) and prohibitive training times, leading to its premature abandonment.

## Run Guide

### Requirements
Ensure you have Python installed, then install the dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### How to Run
To train the agent and run the demo loop, use the following command from the root of the repository:
```bash
PYTHONPATH=src python -m virtual_world_explorer.main
```
Alternatively, you can run the entry script directly:
```bash
python src/virtual_world_explorer/main.py
```

## Component Guide

- **`src/virtual_world_explorer/main.py`**
  The core entry point. Orchestrates the actor-critic DDPG training loop and the interactive 3D graphical demo. It manages the batched background inference for OWL-ViT and provides continuous momentum-based obstacle avoidance logic.
  
- **`src/virtual_world_explorer/render.py`**
  The 3D OpenGL renderer. It implements perspective and orthographic dual-cameras, environment lighting, depth buffering, and rendering of textured 3D models. It also captures the 360-degree egocentric views for the AI's vision system.
  
- **`src/virtual_world_explorer/model3d.py`**
  Handles loading 3D assets (OBJ/MTL files) via `trimesh` and drawing them using raw OpenGL immediate mode. It automatically parses textures, applies materials, and normalizes models to a Z-up coordinate system.
  
- **`src/virtual_world_explorer/env.py`**
  Defines the continuous spatial environment (`GridWorldEnv`). Manages scene objects, continuous step updates utilizing a unicycle model `(v, w)`, and collision detection. The observation state space is heavily optimized for continuous RL.
  
- **`src/virtual_world_explorer/agent.py`**
  Implements the continuous actor-critic model (DDPG) with `ContinuousPolicyNetwork`, `CriticNetwork`, and a `ReplayBuffer`. Optimizes unicycle control using the Bellman loss and Gaussian noise for exploration.
  
- **`src/virtual_world_explorer/detector.py`**
  A foundational abstract semantic sensor that yields normalized directions towards a target within a defined continuous vision radius.
  
- **`src/virtual_world_explorer/owl_vision.py`**
  Integrates zero-shot object detection with HuggingFace's `google/owlvit-base-patch32`. It operates via a batched 360-degree multiview pipeline, efficiently eliminating distractors and determining the precise global target direction, aided by aggressive positional caching to eliminate latency.
