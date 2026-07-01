# Semantic Room Explorer - continuous_env_8d

This branch expands discrete navigation to 8 directions within the continuous environment to approximate continuous kinematics. Coupled with an 8-camera batched OWL-ViT setup, this serves as the final production architecture.

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
  The entry point. Orchestrates the training phase (`train_agent`) and the visual demo loop (`run_demo`). It handles the Batched OWL-ViT inference in a background thread and features a positional cache to minimize inference latency during exploration.
  
- **`src/virtual_world_explorer/render.py`**
  The core of the 3D OpenGL engine. Implements a dual-camera system (an egocentric view for the AI and a 55-degree tilted perspective view for the user). Features depth testing, transparency, material support, and dynamic scaling for 3D models.
  
- **`src/virtual_world_explorer/model3d.py`**
  A bridge between the `trimesh` library and OpenGL immediate mode. It loads OBJ models with MTL materials and PNG textures, adjusts the coordinate system from Y-up to Z-up, and manages rendering.
  
- **`src/virtual_world_explorer/env.py`**
  Defines the continuous RL environment (`GridWorldEnv`). Handles logic for collisions, rewards, and constructs the 11-dimensional observation state representing radar and semantic signals.
  
- **`src/virtual_world_explorer/agent.py`**
  Implements the Actor-Critic DDPG network with Gaussian noise exploration. Optimizes policies based on continuous movement.
  
- **`src/virtual_world_explorer/detector.py`**
  A semantic sensor that provides normalized directional vectors to the target, abstracting the raw visual data.
  
- **`src/virtual_world_explorer/owl_vision.py`**
  Integrates `google/owlvit-base-patch32` to perform Zero-Shot Object Detection. Uses an 8-camera 360° setup that scans all directions simultaneously in a batch, drastically improving visual reasoning and target acquisition.
