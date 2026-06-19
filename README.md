# Virtual World Explorer

Minimal GFX-only prototype for a reinforcement learning agent that navigates a 2D world.

## What it does

- Renders a small 2D grid with OpenGL.
- Trains a tiny Q-learning agent to reach a target object.
- Uses a minimal semantic detector interface to expose the target label to the policy.

## Run

1. Install dependencies from `requirements.txt`.
2. Run `PYTHONPATH=src python -m virtual_world_explorer.main`.

The code is intentionally small and split into a few focused modules.

## How to verify it works

- A GLFW/OpenGL window should open and show a 2D grid.
- You should see one white agent square, one green target object, and two distractor objects with different colors.
- The agent should move toward the target after training finishes.
- In the terminal, you should also see training progress printed every 50 episodes.

If the window does not open, the usual causes are missing OpenGL/GLFW system packages or running in an environment without a display server.
