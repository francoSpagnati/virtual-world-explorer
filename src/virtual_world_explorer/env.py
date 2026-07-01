from __future__ import annotations

from dataclasses import dataclass
import random
import math

from .detector import SemanticDetector

Action = int
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


@dataclass(frozen=True)
class SceneObject:
    label: str
    x: float  
    y: float  
    color: tuple[float, float, float]
    radius: float = 0.35 

class GridWorldEnv: 
    def __init__(self, size: int = 7, seed: int = 7, detector: SemanticDetector | None = None) -> None:
        self.size = float(size) 
        self.random = random.Random(seed)
        self.detector = detector or SemanticDetector()
        self.target_label = "chair"
        
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_radius = 0.25  
        self.step_size = 0.25 
        self.objects: list[SceneObject] = []

    def reset(self) -> tuple[float, float, float, float, float, float, float]:
        object_specs = [
            ("chair", (0.1, 0.7, 0.2)),
            ("table", (0.2, 0.4, 0.9)),
            ("lamp", (0.9, 0.7, 0.2)),
            ("table", (0.2, 0.4, 0.9)),
            ("lamp", (0.9, 0.7, 0.2)),
            ("table", (0.2, 0.4, 0.9)),
            ("lamp", (0.9, 0.7, 0.2)),
        ]
        
        positions = self._sample_continuous_positions(len(object_specs) + 1)
        self.agent_x, self.agent_y = positions[0]
        
        self.objects = [
            SceneObject(label=label, x=pos[0], y=pos[1], color=color)
            for (label, color), pos in zip(object_specs, positions[1:])
        ]
        return self._observation()

    def step(self, action: Action) -> tuple[tuple[float, float, float, float, float, float, float], float, bool, dict[str, object]]:
        target = self._target_object()
        prev_dist = math.hypot(self.agent_x - target.x, self.agent_y - target.y)

        next_x, next_y = self.agent_x, self.agent_y
        if action == UP:    next_y -= self.step_size
        elif action == DOWN:  next_y += self.step_size
        elif action == LEFT:  next_x -= self.step_size
        elif action == RIGHT: next_x += self.step_size

        next_x = max(self.agent_radius, min(self.size - self.agent_radius, next_x))
        next_y = max(self.agent_radius, min(self.size - self.agent_radius, next_y))

        hit_obstacle = False
        for obj in self.objects:
            if obj.label != self.target_label:
                dist_to_obj = math.hypot(next_x - obj.x, next_y - obj.y)
                if dist_to_obj < (self.agent_radius + obj.radius):
                    hit_obstacle = True
                    break

        if hit_obstacle:
            reward = -1.0
            done = False
        else:
            self.agent_x, self.agent_y = next_x, next_y
            curr_dist = math.hypot(self.agent_x - target.x, self.agent_y - target.y)
            
            done = curr_dist < (self.agent_radius + target.radius)
            
            reward = -0.05 + 0.1 * (prev_dist - curr_dist)
            if done:
                reward = 2.0

        info = {"target_label": target.label, "target_position": (target.x, target.y)}
        return self._observation(), reward, done, info

    def _observation(self) -> tuple[float, float, float, float, float, float, float]:
        target = self._target_object()
        dist = math.hypot(self.agent_x - target.x, self.agent_y - target.y)
        
        if dist <= 3.5:
            dx = (target.x - self.agent_x) / dist
            dy = (target.y - self.agent_y) / dist
            visible = 1.0
        else:
            dx, dy, visible = 0.0, 0.0, 0.0

        danger_up = float(self._check_collision_at(self.agent_x, self.agent_y - self.step_size))
        danger_down = float(self._check_collision_at(self.agent_x, self.agent_y + self.step_size))
        danger_left = float(self._check_collision_at(self.agent_x - self.step_size, self.agent_y))
        danger_right = float(self._check_collision_at(self.agent_x + self.step_size, self.agent_y))

        return (dx, dy, visible, danger_up, danger_down, danger_left, danger_right)

    def _check_collision_at(self, x: float, y: float) -> bool:
        if x < self.agent_radius or x > (self.size - self.agent_radius): return True
        if y < self.agent_radius or y > (self.size - self.agent_radius): return True
        for obj in self.objects:
            if obj.label != self.target_label:
                if math.hypot(x - obj.x, y - obj.y) < (self.agent_radius + obj.radius):
                    return True
        return False

    def _target_object(self) -> SceneObject:
        for scene_object in self.objects:
            if scene_object.label == self.target_label:
                return scene_object
        raise RuntimeError("Target object is missing from the scene")

    def _sample_continuous_positions(self, count: int) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        min_safety_dist = 0.9
        while len(positions) < count:
            candidate = (self.random.uniform(0.5, self.size - 0.5), self.random.uniform(0.5, self.size - 0.5))
            if all(math.hypot(candidate[0] - p[0], candidate[1] - p[1]) >= min_safety_dist for p in positions):
                positions.append(candidate)
        return positions