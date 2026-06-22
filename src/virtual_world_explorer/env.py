from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence, cast

from .detector import Detection, SemanticDetector


Action = int
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


@dataclass(frozen=True)
class SceneObject:
    label: str
    x: int
    y: int
    color: tuple[float, float, float]


class GridWorldEnv:
    def __init__(self, size: int = 7, seed: int = 7, detector: SemanticDetector | None = None) -> None:
        self.size = size
        self.random = random.Random(seed)
        self.detector = detector or SemanticDetector()
        self.target_label = "chair"
        self.agent_x = 0
        self.agent_y = 0
        self.objects: list[SceneObject] = []

    def reset(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        positions = self._sample_positions(4)
        self.agent_x, self.agent_y = positions[0]
        object_specs = [
            ("chair", (0.1, 0.7, 0.2)),
            ("table", (0.2, 0.4, 0.9)),
            ("lamp", (0.9, 0.7, 0.2)),
        ]
        self.objects = [
            SceneObject(label=label, x=position[0], y=position[1], color=color)
            for (label, color), position in zip(object_specs, positions[1:])
        ]
        return self._observation()

    def step(self, action: Action) -> tuple[tuple[int, int, int, int, int, int, int, int, int], float, bool, dict[str, object]]:
        target = self._target_object()
        previous_distance = self._manhattan_distance(self.agent_x, self.agent_y, target.x, target.y)
        previous_position = (self.agent_x, self.agent_y)

        new_x, new_y = self.agent_x, self.agent_y
        if action == UP:
            new_y = max(0, self.agent_y - 1)
        elif action == DOWN:
            new_y = min(self.size - 1, self.agent_y + 1)
        elif action == LEFT:
            new_x = max(0, self.agent_x - 1)
        elif action == RIGHT:
            new_x = min(self.size - 1, self.agent_x + 1)

        hit_distractor = any(
            obj.label != self.target_label and new_x == obj.x and new_y == obj.y
            for obj in self.objects
        )

        if hit_distractor:
            reward = -1.0
            done = False
        else:
            self.agent_x, self.agent_y = new_x, new_y
            current_distance = self._manhattan_distance(self.agent_x, self.agent_y, target.x, target.y)
            done = self.agent_x == target.x and self.agent_y == target.y
            reward = -0.01 + 0.02 * (previous_distance - current_distance)
            if previous_position == (self.agent_x, self.agent_y):
                reward -= 0.03
            if done:
                reward = 1.0

        info = {"target_label": target.label, "target_position": (target.x, target.y)}
        return self._observation(), reward, done, info

    def _observation(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        detection = self.detector.detect(cast(list[object], self.objects), (self.agent_x, self.agent_y), self.target_label)
        danger_up = int(any(
            obj.label != self.target_label and obj.x == self.agent_x and obj.y == self.agent_y - 1
            for obj in self.objects
        ))
        danger_down = int(any(
            obj.label != self.target_label and obj.x == self.agent_x and obj.y == self.agent_y + 1
            for obj in self.objects
        ))
        danger_left = int(any(
            obj.label != self.target_label and obj.x == self.agent_x - 1 and obj.y == self.agent_y
            for obj in self.objects
        ))
        danger_right = int(any(
            obj.label != self.target_label and obj.x == self.agent_x + 1 and obj.y == self.agent_y
            for obj in self.objects
        ))
        return (
            self.agent_x, self.agent_y,
            detection.dx, detection.dy, int(detection.visible),
            danger_up, danger_down, danger_left, danger_right,
        )

    def _target_object(self) -> SceneObject:
        for scene_object in self.objects:
            if scene_object.label == self.target_label:
                return scene_object
        raise RuntimeError("Target object is missing from the scene")

    def _sample_positions(self, count: int) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        while len(positions) < count:
            candidate = (self.random.randrange(self.size), self.random.randrange(self.size))
            if candidate not in positions:
                positions.append(candidate)
        return positions

    @staticmethod
    def _manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)
