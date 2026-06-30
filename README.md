# Semantic Room Explorer - 3d_settings

This branch migrates the visualization to a perspective 3D space using glFrustum, depth testing and hardware lighting. It renders the agent as a 3D cube and objects as meshes without altering the underlying Reinforcement Learning logic.

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

This branch incorporates a full 3D rendering pipeline and advanced vision logic on top of the original Reinforcement Learning baseline. Here is a synthesized summary of the core components:

*   **`main.py`**: The central entry point orchestrating both the training loop (`train_agent`) and the visual demo loop (`run_demo`). It spawns a background thread for OWL-ViT inference to keep performance high and introduces a continuous momentum logic to avoid stalling on collisions.
*   **`render.py`**: The core of the 2D to 3D transition. It replaces orthographic top-down views with a `glFrustum` perspective camera, implements hardware lighting (`GL_LIGHT0`), depth testing (`GL_DEPTH_TEST`), and alpha blending. It coordinates the drawing of the 3D grid, the agent's cube/mesh, and other objects.
*   **`model3d.py`**: A specialized loader that delegates parsing of `.obj` models, `.mtl` materials, and textures to `trimesh`, but manages the actual rendering directly in OpenGL immediate mode. It handles Y-up to Z-up conversion and bounds scaling automatically.
*   **`env.py`**: The continuous GridWorld environment. It models coordinates continuously (no longer discrete grids), handles collisions, and evaluates continuous actions `(v, w)` from the agent, dispensing rewards and calculating the 11-dimensional observation state.
*   **`agent.py`**: Implements a Continuous Policy Network (actor-critic architecture) using a Replay Buffer. It outputs continuous control parameters (linear and angular velocities) bound by `tanh` and explores using gaussian noise.
*   **`detector.py`**: A baseline semantic sensor that checks if a target object is within the agent's visible radius, returning its angle for the observation vector.
*   **`owl_vision.py`**: Replaces the baseline sensor with a zero-shot `google/owlvit-base-patch32` object detector. It performs a batched inference over 8 simultaneous perspectives (360-degree vision) to find the "chair" among distractors like tables and lamps, heavily optimized using a positional cache.
