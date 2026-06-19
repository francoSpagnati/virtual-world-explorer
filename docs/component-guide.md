# Component Guide

This file explains every piece of the minimal prototype and why it exists.

## Overall flow

1. `main.py` trains the agent.
2. `GridWorldEnv` generates a tiny 2D scene.
3. `SemanticDetector` reads the scene and returns a semantic hint about the target object.
4. `QLearningAgent` uses that hint plus its current position to choose a move.
5. `OpenGLRenderer` draws the final scene so you can see the agent navigate.

The design is intentionally simple. The code is small enough to read top to bottom without jumping across many abstractions.

## `requirements.txt`

This file lists the only external runtime dependencies.

- `numpy` is kept available for future extensions, even if the current prototype does not need heavy numeric work.
- `PyOpenGL` provides the GFX layer.
- `glfw` creates the OpenGL window and handles the event loop.

## `src/virtual_world_explorer/env.py`

This is the environment.

### `SceneObject`

Represents one object inside the world.

- `label` is the semantic class, such as `chair` or `table`.
- `x` and `y` are the grid coordinates.
- `color` is only for visualization.

### `GridWorldEnv`

Owns the grid, the agent position, and the scene objects.

- `reset()` creates a new scene with one target object and two distractors.
- `step(action)` moves the agent and returns the next observation, reward, and termination flag.
- `_observation()` builds the state that the RL agent sees.
- `_target_object()` finds the target object by label.
- `_sample_positions()` keeps object placement simple and collision-free.

### Reward logic

The reward is intentionally minimal.

- Each step costs a small negative reward.
- Moving closer to the target adds a small shaping bonus.
- Reaching the target gives a large positive reward.

This keeps training simple while still encouraging goal-directed behavior.

## `src/virtual_world_explorer/detector.py`

This is the semantic detection layer.

### `Detection`

Stores the result of the semantic detector.

- `label` is the recognized class.
- `dx` and `dy` are the coarse direction toward the target.
- `visible` says whether the target is within the detector range.

### `SemanticDetector`

This is the minimal stand-in for a real vision-language detector such as OWL.

- `detect()` looks for the target object in the scene.
- If the target is close enough, it returns a direction hint.
- If the target is out of range, it returns `visible=False` and zeros for the direction.

In a more realistic version, this module would be the place where image crops or billboards are passed into OWL or a similar model.

## `src/virtual_world_explorer/agent.py`

This is the RL agent.

### `QLearningAgent`

The project uses tabular Q-learning because it is the smallest training setup that still counts as reinforcement learning.

- `choose_action()` implements epsilon-greedy exploration.
- `learn()` updates the Q-table from the reward signal.
- `decay_exploration()` slowly reduces randomness so the policy becomes more stable over time.

The state is small on purpose: agent position, coarse direction to the target, and a visibility flag.

## `src/virtual_world_explorer/render.py`

This file handles the GFX output.

### `RendererConfig`

Small configuration object for window size and cell padding.

### `OpenGLRenderer`

Draws the scene with OpenGL.

- `initialize()` creates the GLFW window and sets up the 2D orthographic projection.
- `draw()` clears the screen and draws the grid, objects, and agent.
- `should_close()` lets the main loop stop cleanly.
- `poll()` processes window events.
- `shutdown()` releases OpenGL and GLFW resources.

The renderer is intentionally plain: no shader pipeline, no engine, no framework abstraction.

## `src/virtual_world_explorer/main.py`

This is the entry point.

### `train_agent()`

Runs a short Q-learning loop.

- Resets the environment.
- Chooses actions.
- Applies Q-learning updates.
- Prints progress every 50 episodes.

### `run_demo()`

Runs a short visual demo after training.

- Temporarily disables exploration.
- Lets the agent follow the learned policy.
- Renders each step with OpenGL.

### `main()`

The top-level orchestration function.

1. Train the agent.
2. Show the trained policy in the OpenGL window.

## Why this structure is minimal

The project is split only by responsibility, not by abstraction layers.

- Environment logic stays in one place.
- Detection logic stays in one place.
- Learning logic stays in one place.
- Rendering logic stays in one place.

That keeps the code easy to inspect and easy to replace later if you want a real OWL-based detector or a richer scene.
