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

This project is modularly structured into the following core components:

- **`main.py`**: The central orchestrator. It executes the RL training (`train_agent`), runs the visual demo (`run_demo`), coordinates background inference for computer vision, and applies movement heuristics to prevent looping.
- **`render.py`**: The 3D visual frontend. Uses PyOpenGL to render a 3D perspective with lighting, textured OBJ models for the agent and obstacles, and an orthographic HUD. It also captures the 360° visual frames used by the AI.
- **`model3d.py`**: A dedicated 3D utility module that leverages `trimesh` to load 3D assets (OBJ, MTL) and their textures, mapping them seamlessly into the immediate-mode OpenGL scene.
- **`env.py`**: The continuous `GridWorldEnv` environment. Manages spatial coordinates, obstacle collision logic, rewards, and assembles the 11-dimensional observation state for the agent.
- **`agent.py`**: The RL agent utilizing a continuous policy. Contains Actor and Critic neural networks along with a Replay Buffer to learn continuous velocity and angular controls (`v, w`).
- **`detector.py`**: The foundational semantic sensor logic, determining the relative target direction.
- **`owl_vision.py`**: The zero-shot visual detection pipeline. Utilizes Google's OWL-ViT to perform batched 360° object detection, allowing the agent to locate the target purely from rendered camera views.
