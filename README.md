# Semantic Room Explorer - main

This branch establishes the 2D grid foundation using PyOpenGL and GLFW. It introduces a tabular Q-learning agent navigating a 7x7 arena via simulated relative semantic sensors and foundational anti-loop reward mechanics.

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
  The central orchestrator. It executes the RL training (`train_agent`), runs the visual demo (`run_demo`), coordinates background inference for computer vision, and applies movement heuristics to prevent looping.

- **`src/virtual_world_explorer/render.py`**
  The 3D visual frontend. Uses PyOpenGL to render a 3D perspective with lighting, textured OBJ models for the agent and obstacles, and an orthographic HUD. It also captures the 360° visual frames used by the AI.

- **`src/virtual_world_explorer/model3d.py`**
  A dedicated 3D utility module that leverages `trimesh` to load 3D assets (OBJ, MTL) and their textures, mapping them seamlessly into the immediate-mode OpenGL scene.

- **`src/virtual_world_explorer/env.py`**
  The continuous `GridWorldEnv` environment. Manages spatial coordinates, obstacle collision logic, rewards, and assembles the 11-dimensional observation state for the agent.

- **`src/virtual_world_explorer/agent.py`**
  The RL agent utilizing a continuous policy. Contains Actor and Critic neural networks along with a Replay Buffer to learn continuous velocity and angular controls (`v, w`).

- **`src/virtual_world_explorer/detector.py`**
  The foundational semantic sensor logic, determining the relative target direction.

- **`src/virtual_world_explorer/owl_vision.py`**
  The zero-shot visual detection pipeline. Utilizes Google's OWL-ViT to perform batched 360° object detection, allowing the agent to locate the target purely from rendered camera views.
