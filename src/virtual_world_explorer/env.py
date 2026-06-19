from __future__ import annotations

from dataclasses import dataclass
import random

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

    def reset(self) -> tuple[int, int, int, int, int]:
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

    def step(self, action: Action) -> tuple[tuple[int, int, int, int, int], float, bool, dict[str, object]]:
        target_before = self._target_object()
        previous_distance = self._manhattan_distance(self.agent_x, self.agent_y, target_before.x, target_before.y)
        previous_position = (self.agent_x, self.agent_y)

        if action == UP:
            self.agent_y = max(0, self.agent_y - 1)
        elif action == DOWN:
            self.agent_y = min(self.size - 1, self.agent_y + 1)
        elif action == LEFT:
            self.agent_x = max(0, self.agent_x - 1)
        elif action == RIGHT:
            self.agent_x = min(self.size - 1, self.agent_x + 1)

        target_after = self._target_object()
        current_distance = self._manhattan_distance(self.agent_x, self.agent_y, target_after.x, target_after.y)
        done = self.agent_x == target_after.x and self.agent_y == target_after.y
        reward = -0.01 + 0.02 * (previous_distance - current_distance)
        if previous_position == (self.agent_x, self.agent_y):
            reward -= 0.03
        if done:
            reward = 1.0

        info = {"target_label": target_after.label, "target_position": (target_after.x, target_after.y)}
        return self._observation(), reward, done, info

    def _observation(self) -> tuple[int, int, int, int, int]:
        detection = self.detector.detect(self.objects, (self.agent_x, self.agent_y), self.target_label)
        return (self.agent_x, self.agent_y, detection.dx, detection.dy, int(detection.visible))

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
