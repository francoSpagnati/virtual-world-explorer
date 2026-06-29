from __future__ import annotations

from dataclasses import dataclass
import random
import math
import numpy

from .detector import SemanticDetector

# Rimosse le vecchie costanti discrete UP, DOWN, LEFT, RIGHT

@dataclass(frozen=True)
class SceneObject:
    label: str
    x: float  # Coordinate continue
    y: float  # Coordinate continue
    color: tuple[float, float, float]
    radius: float = 0.35  # Raggio fisico di ingombro dell'ostacolo


class GridWorldEnv:
    def __init__(self, size: int = 7, seed: int = 7, detector: SemanticDetector | None = None) -> None:
        self.size = float(size)  # L'arena è uno spazio continuo float
        self.random = random.Random(seed)
        self.detector = detector or SemanticDetector()
        self.target_label = "chair"
        
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_radius = 0.25  # Raggio fisico dell'agente cubo
        
        # Parametri di movimento continuo massimi per singolo step
        self.max_linear_velocity = 0.25 
        self.max_angular_velocity = math.radians(45.0) # Massimo 45 gradi di rotazione per step
        self.agent_heading = 0.0 # Angolo di orientamento corrente dell'agente in radianti
        
        self.objects: list[SceneObject] = []

    def reset(self) -> tuple[float, ...]:
        object_specs = [
            ("chair", (0.1, 0.8, 0.1)),
            ("table", (0.8, 0.1, 0.1)),
            ("lamp", (0.1, 0.1, 0.8)),
        ]
        
        positions = self._sample_continuous_positions(len(object_specs) + 1)
        
        self.agent_x, self.agent_y = positions[0]
        self.agent_heading = self.random.uniform(0, 2 * math.pi) # Orientamento iniziale casuale
        
        self.objects = [
            SceneObject(label=spec[0], x=pos[0], y=pos[1], color=spec[1])
            for spec, pos in zip(object_specs, positions[1:])
        ]
        return self._get_obs()

    def step(self, action: numpy.ndarray | list[float]) -> tuple[tuple[float, ...], float, bool, dict]:
        """
        Esegue un passo nello spazio di controllo continuo.
        action: [v, w] dove:
          - v: velocità lineare scalata tra [-1, 1] -> mappata su [0, max_linear_velocity]
          - w: velocità angolare scalata tra [-1, 1] -> mappata su [-max_angular_velocity, max_angular_velocity]
        """
        v_input, w_input = action[0], action[1]
        
        # Denormalizzazione degli input continui della rete neurale
        linear_vel = max(0.0, ((v_input + 1.0) / 2.0) * self.max_linear_velocity) # solo movimento in avanti
        angular_vel = w_input * self.max_angular_velocity
        
        # Aggiorna l'orientamento dell'agente
        self.agent_heading = (self.heading + angular_vel) % (2 * math.pi)
        
        # Calcola la traiettoria dello spostamento continuo nello spazio cartesiano
        next_x = self.agent_x + linear_vel * math.cos(self.agent_heading)
        next_y = self.agent_y + linear_vel * math.sin(self.agent_heading)
        
        # Controllo delle collisioni continue prima di convalidare la posizione
        if not self._check_collision_at(next_x, next_y):
            self.agent_x = next_x
            self.agent_y = next_y
            collision = False
        else:
            collision = True

        target = self._target_object()
        distance = math.hypot(self.agent_x - target.x, self.agent_y - target.y)
        
        # Calcolo del sistema di Reward continuo
        done = distance < (self.agent_radius + target.radius)
        if done:
            reward = 150.0
        elif collision:
            reward = -2.5
        else:
            # Reward shaping basata sulla vicinanza progressiva
            reward = -0.1 - (distance * 0.1)

        return self._get_obs(), reward, done, {}

    @property
    def heading(self) -> float:
        return self.agent_heading

    def _get_obs(self) -> tuple[float, ...]:
        """
        Restituisce lo stato osservato adattandosi al SemanticDetector originale.
        Vettore esteso a 11 elementi per supportare la granularità a 8 telecamere:
        [dx, dy, visible, danger_0, danger_45, danger_90, danger_135, danger_180, danger_225, danger_270, danger_315]
        """
        target = self._target_object()
        # Otteniamo il risultato dal detector originale
        detector_result = self.detector.detect(self.objects, (self.agent_x, self.agent_y), self.target_label)
        
        # Estraiamo i valori in base alla struttura nativa del tuo SemanticDetector
        visible = float(detector_result.visible)
        
        # Nel detector originale dx e dy sono attributi diretti dell'oggetto o ricavabili.
        # Per massima sicurezza e compatibilità, calcoliamo direttamente il vettore direzionale geometrico continuo:
        dx, dy = 0.0, 0.0
        if detector_result.visible:
            # Calcolo continuo verso il target
            dist = math.hypot(target.x - self.agent_x, target.y - self.agent_y)
            if dist > 0:
                global_target_angle = math.atan2(target.y - self.agent_y, target.x - self.agent_x)
                relative_angle = global_target_angle - self.agent_heading
                dx = math.cos(relative_angle)
                dy = math.sin(relative_angle)

        # Radar di prossimità a 8 canali per la navigazione continua omnidirezionale
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