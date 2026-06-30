# Semantic Room Explorer - 3d_horizontal_camera

This branch implements a dual-camera pipeline, with a stable tilted perspective view for user monitoring and a 4-directional egocentric view for the agent. This restricts visibility to a tight radius to optimize visual object detection.

## Run Guide

### Requirements
Ensure you have the necessary dependencies installed. You can install them via pip:

```bash
pip install -r requirements.txt
```

This will install packages such as `PyOpenGL`, `glfw`, `Pillow`, `numpy`, `trimesh`, `torch`, and `transformers`.

### How to Run
To run the training and start the simulation demo, execute the main entry point:

```bash
python src/virtual_world_explorer/main.py
```

## Component Guide (Synthesized)

The **Virtual World Explorer 3D** trains an agent to navigate a 3D environment to find a target while avoiding obstacles. It employs Reinforcement Learning (Q-Learning) alongside a visual perception module (OWL-ViT) for zero-shot object detection.

### Architecture Overview

- **`src/virtual_world_explorer/main.py`**
  The entry point of the application. It runs the training loop, manages the 3D visual demo, and handles batched inferences for zero-shot object detection.

- **`src/virtual_world_explorer/render.py`**
  The core 3D rendering module using PyOpenGL. In this branch, it implements a dual-camera pipeline. One is a stable tilted perspective view for the user, and the other is a 4-directional egocentric view for the agent, which optimizes visibility and restricts the view radius for visual object detection.

- **`src/virtual_world_explorer/model3d.py`**
  Handles 3D asset management. Uses `trimesh` to load OBJ files and their MTL textures into the immediate-mode OpenGL rendering pipeline.

- **`src/virtual_world_explorer/env.py`**
  The RL environment mapping discrete grid cells to 3D space.

- **`src/virtual_world_explorer/agent.py`**
  Implements the Q-Learning agent managing discrete 4D actions and building the Q-table.

- **`src/virtual_world_explorer/detector.py`**
  Simulated semantic sensor for training logic.

- **`src/virtual_world_explorer/owl_vision.py`**
  Integrates `google/owlvit-base-patch32` for 360-degree batched visual detection from the egocentric cameras.
