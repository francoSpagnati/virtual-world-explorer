# Semantic Room Explorer - continuous_mov

This branch evaluates a continuous unicycle model (v, w) for agent control. The expanded state space caused severe policy degeneration (spinning local minima) and prohibitive training times, leading to its premature abandonment.

## Run Guide

**Requirements:**
Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```
*(Dependencies include PyOpenGL, glfw, Pillow, numpy, trimesh, torch, and transformers).*

**How to run it:**
To run the project, which trains the DDPG agent and subsequently launches the 3D OpenGL demonstration with the OWL-ViT vision system, execute:
```bash
python -m src.virtual_world_explorer.main
```

## Component Guide

*   **`main.py`**: The core entry point. Orchestrates the actor-critic DDPG training loop and the interactive 3D graphical demo. It manages the batched background inference for OWL-ViT and provides continuous momentum-based obstacle avoidance logic.
*   **`render.py`**: The 3D OpenGL renderer. It implements perspective and orthographic dual-cameras, environment lighting, depth buffering, and rendering of textured 3D models. It also captures the 360-degree egocentric views for the AI's vision system.
*   **`model3d.py`**: Handles loading 3D assets (OBJ/MTL files) via `trimesh` and drawing them using raw OpenGL immediate mode. It automatically parses textures, applies materials, and normalizes models to a Z-up coordinate system.
*   **`env.py`**: Defines the continuous spatial environment (`GridWorldEnv`). Manages scene objects, continuous step updates utilizing a unicycle model `(v, w)`, and collision detection. The observation state space is heavily optimized for continuous RL.
*   **`agent.py`**: Implements the continuous actor-critic model (DDPG) with `ContinuousPolicyNetwork`, `CriticNetwork`, and a `ReplayBuffer`. Optimizes unicycle control using the Bellman loss and Gaussian noise for exploration.
*   **`detector.py`**: A foundational abstract semantic sensor that yields normalized directions towards a target within a defined continuous vision radius.
*   **`owl_vision.py`**: Integrates zero-shot object detection with HuggingFace's `google/owlvit-base-patch32`. It operates via a batched 360-degree multiview pipeline, efficiently eliminating distractors and determining the precise global target direction, aided by aggressive positional caching to eliminate latency.
