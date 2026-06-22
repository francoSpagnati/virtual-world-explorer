from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    dx: int
    dy: int
    visible: bool


class SemanticDetector:
    def __init__(self, vision_radius: int = 3) -> None:
        self.vision_radius = vision_radius

    def detect(self, objects: list[object], agent_position: tuple[int, int], target_label: str) -> Detection:
        """Analizza la scena per localizzare l'oggetto target rispetto all'agente."""
        target = None
        for scene_object in objects:
            if getattr(scene_object, "label", None) == target_label:
                target = scene_object
                break

        if target is None:
            raise RuntimeError(f"Target object '{target_label}' is missing from the scene")

        agent_x, agent_y = agent_position
        dx = target.x - agent_x
        dy = target.y - agent_y
        
        # Verifica se l'oggetto rientra nel raggio visivo limitato dell'agente
        visible = abs(dx) <= self.vision_radius and abs(dy) <= self.vision_radius
        if not visible:
            return Detection(label=target.label, dx=0, dy=0, visible=False)

        # Restituisce la direzione normalizzata (-1, 0, 1)
        return Detection(label=target.label, dx=self._sign(dx), dy=self._sign(dy), visible=True)

    @staticmethod
    def _sign(value: int) -> int:
        if value < 0:
            return -1
        if value > 0:
            return 1
        return 0