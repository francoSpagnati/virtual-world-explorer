from __future__ import annotations

from dataclasses import dataclass
import random
import math
import numpy

from .detector import SemanticDetector

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

        self.max_linear_velocity = 0.25 
        self.max_angular_velocity = math.radians(45.0)
        self.agent_heading = 0.0 

        self.objects: list[SceneObject] = []

    def reset(self) -> tuple[float, ...]:
        object_specs = [
            ("chair", (0.1, 0.8, 0.1)),
            ("table", (0.8, 0.1, 0.1)),
            ("lamp", (0.1, 0.1, 0.8)),
        ]
        
        positions = self._sample_continuous_positions(len(object_specs) + 1)
        
        self.agent_x, self.agent_y = positions[0]
        self.agent_heading = self.random.uniform(0, 2 * math.pi) 

        self.objects = [
            SceneObject(label=spec[0], x=pos[0], y=pos[1], color=spec[1])
            for spec, pos in zip(object_specs, positions[1:])
        ]
        return self._get_obs()

    def step(self, action: numpy.ndarray | list[float]) -> tuple[tuple[float, ...], float, bool, dict]:

        v_input, w_input = action[0], action[1]
        
        linear_vel = max(0.0, ((v_input + 1.0) / 2.0) * self.max_linear_velocity) 
        angular_vel = w_input * self.max_angular_velocity
        
        self.agent_heading = (self.heading + angular_vel) % (2 * math.pi)
        
        next_x = self.agent_x + linear_vel * math.cos(self.agent_heading)
        next_y = self.agent_y + linear_vel * math.sin(self.agent_heading)
        
        if not self._check_collision_at(next_x, next_y):
            self.agent_x = next_x
            self.agent_y = next_y
            collision = False
        else:
            collision = True

        target = self._target_object()
        distance = math.hypot(self.agent_x - target.x, self.agent_y - target.y)
        
        done = distance < (self.agent_radius + target.radius)
        if done:
            reward = 150.0
        elif collision:
            reward = -2.5
        else:
            reward = -0.1 - (distance * 0.1)

        return self._get_obs(), reward, done, {}

    @property
    def heading(self) -> float:
        return self.agent_heading

    def _get_obs(self) -> tuple[float, ...]:

        target = self._target_object()
        detector_result = self.detector.detect(self.objects, (self.agent_x, self.agent_y), self.target_label)
        
        visible = float(detector_result.visible)
        
        dx, dy = 0.0, 0.0
        if detector_result.visible:
            dist = math.hypot(target.x - self.agent_x, target.y - self.agent_y)
            if dist > 0:
                global_target_angle = math.atan2(target.y - self.agent_y, target.x - self.agent_x)
                relative_angle = global_target_angle - self.agent_heading
                dx = math.cos(relative_angle)
                dy = math.sin(relative_angle)

        danger_signals = []
        for i in range(8):
            angle = self.agent_heading + math.radians(i * 45.0)
            check_x = self.agent_x + self.max_linear_velocity * math.cos(angle)
            check_y = self.agent_y + self.max_linear_velocity * math.sin(angle)
            danger_signals.append(float(self._check_collision_at(check_x, check_y)))

        return (dx, dy, visible, *danger_signals)

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