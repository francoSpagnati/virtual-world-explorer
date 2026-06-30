# Semantic Room Explorer - continuous_env

## Description
This branch replaces the tile grid with a continuous coordinate space while keeping discrete 4D actions. It introduces continuous bounding-box collision handling and strictly normalizes state vectors for neural network stability.

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

## Component Guide Summary

Here is a synthesized summary of the system architecture and its core modules based on the component guide:

- **`main.py`**: The entry point that orchestrates the RL training (`train_agent`) and 3D visual demo (`run_demo`). It handles batched OWL-ViT inference threading and predictive action maneuvering.
- **`render.py`**: The core of the 3D transition. Uses OpenGL to manage perspective/orthographic cameras, lighting (`GL_LIGHT0`), depth testing, and renders 3D models alongside the HUD overlay.
- **`model3d.py`**: Utilizes `trimesh` to load 3D objects (OBJ) and materials (MTL/textures), handling coordinate conversions (Y-up to Z-up) and rendering them in OpenGL immediate mode.
- **`env.py`**: Defines the continuous `GridWorldEnv`. Handles randomized continuous placements, normalized 11-D observation vectors (omitting absolute agent coordinates for translational invariance), and continuous bounding-box collision logic.
- **`agent.py`**: The RL brain (Continuous Policy Network / DDPG), comprising an Actor network for continuous action outputs `(v, w)`, a Critic network, and a Replay Buffer for learning.
- **`detector.py`**: A baseline semantic sensor providing relative normalized directions to target objects within a visibility radius.
- **`owl_vision.py`**: Integrates Deep Learning zero-shot object detection (`OWL-ViT`). Implements a 360° multiview pipeline with batched inference and positional caching to provide instant global direction tracking without movement lag.
