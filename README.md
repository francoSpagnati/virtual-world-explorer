# Semantic Room Explorer - 3d_horizontal_camera

This branch implements a dual-camera pipeline, with a stable tilted perspective view for user monitoring and a 4-directional egocentric view for the agent. This restricts visibility to a tight radius to optimize visual object detection.

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
