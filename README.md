# Semantic Room Explorer - 3d_parametric

This branch decouples the system from hardcoded grid constraints by scaling all parameters with respect to an arbitrary grid size. It maintains translation invariance within the 7D relative state space to allow zero-shot policy generalization.

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

## Component Guide (Synthesized)

The **Virtual World Explorer 3D** trains an agent (a robot) to navigate a continuous 3D environment to find a target (a chair) while avoiding obstacles (a table and a lamp). It employs Reinforcement Learning (DDPG) alongside a visual perception module (OWL-ViT) for object detection.

### Architecture Overview

- **`src/virtual_world_explorer/main.py`**
  The entry point of the application. It orchestrates the entire flow: running the training loop (`train_agent`), managing the 3D visual demo (`run_demo`), and handling batched inferences for zero-shot object detection to prevent gameplay lag.

- **`src/virtual_world_explorer/render.py`**
  The core 3D rendering module. It utilizes PyOpenGL to provide dual camera perspectives (a fixed user view and an egocentric AI view). It handles depth buffering, lighting, and renders 3D models with textures instead of simple 2D shapes.

- **`src/virtual_world_explorer/model3d.py`**
  Responsible for 3D asset management. It leverages the `trimesh` library to parse OBJ models and MTL materials, scaling them and converting their coordinate systems before rendering them via OpenGL immediate mode.

- **`src/virtual_world_explorer/env.py`**
  The Reinforcement Learning environment (`GridWorldEnv`). It handles continuous spatial coordinates, object placement, collision detection, and reward calculation. It abstracts the world into an 11-dimensional observation state (ignoring absolute coordinates to maintain translation invariance).

- **`src/virtual_world_explorer/agent.py`**
  Implements the continuous reinforcement learning agent (DDPG) utilizing an Actor-Critic architecture. It explores the environment using Gaussian noise and learns from a replay buffer.

- **`src/virtual_world_explorer/detector.py`**
  A semantic sensor that gives the agent directional awareness (abstract orientation) towards the target if it is within a specified vision radius.

- **`src/virtual_world_explorer/owl_vision.py`**
  Integrates a deep learning zero-shot object detection pipeline (`google/owlvit-base-patch32`). It processes a 360-degree batched scan of the environment from the agent's perspective, mapping objects in the field of view without hardcoded rules, backed by positional caching for performance.
