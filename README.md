# Virtual World Explorer

Minimal GFX-only prototype for a reinforcement learning agent that navigates a 3D world.

## What it does

- Renders a 7×7 grid with 3D models (OBJ) using raw OpenGL immediate mode — no high-level frameworks.
- Parses OBJ + MTL files manually (Chair with texture, Lamp and Table with vertex colors).
- Trains a tabular Q-learning agent to reach a target object (chair) while avoiding distractors (table, lamp).
- Uses a semantic detector interface to expose the target label to the policy.
- Lighting with GL_LIGHT0, smooth shading, depth testing.

## Run

1. Install dependencies from `requirements.txt`.
2. Extract OBJ files from assets/Chair.zip, assets/Lamp.zip, assets/Table.zip into `assets/models/`.
3. Run `PYTHONPATH=src python -m virtual_world_explorer.main`.

The code is intentionally small and split into a few focused modules.

## How to verify it works

- Training prints reward + epsilon every 50 episodes (5000 total).
- A GLFW/OpenGL window opens showing a 3D perspective grid with:
  - White agent cube
  - 3D chair model (textured) — target
  - 3D lamp model (colored) — distractor
  - 3D table model (colored) — distractor
- The agent moves toward the chair after training.
- Terminal prints "Target reached, resetting scene." on each success.

If the window does not open, the usual causes are missing OpenGL/GLFW system packages or running in an environment without a display server.
