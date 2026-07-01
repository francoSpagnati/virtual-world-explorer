from __future__ import annotations

from dataclasses import dataclass
import math
import os

import glfw
from typing import Any
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINES,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_TRIANGLES,
    glDisable,
    glEnable,
    glBegin,
    glClear,
    glPixelStorei,
    GL_PACK_ALIGNMENT,
    glClearColor,
    glColor3f,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glRectf,
    glVertex2f,
    glVertex3f,
    glViewport,
    glFrustum,
    glRotatef,
    glTranslatef,
    glScalef,
    glPushMatrix,
    glPopMatrix,
    glNormal3f,
    GL_TEXTURE_2D,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_LINEAR,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    glBindTexture,
    glGenTextures,
    glTexImage2D,
    glTexParameterf,
    glTexCoord2f,
    glBlendFunc,
    glReadPixels,
    GL_RGB,
    GL_LIGHTING,
    GL_LIGHT0,
    GL_POSITION,
    GL_AMBIENT,
    GL_DIFFUSE,
    GL_SPECULAR,
    GL_NORMALIZE,
    GL_DIFFUSE,
    GL_SPECULAR,
    GL_FRONT_AND_BACK,
    GL_AMBIENT_AND_DIFFUSE,
    glMaterialfv,
    glLightfv,
    GL_FLAT,
    GL_SMOOTH,
    glShadeModel,
    GL_FRONT,
)

from PIL import Image
import numpy as np

from .env import GridWorldEnv
from .model3d import Model3D


@dataclass(frozen=True)
class RendererConfig:
    window_size: int = 720
    cell_padding: float = 0.08
    hud_margin: float = 0.18
    hud_line_height: float = 11.0
    hud_scale: float = 1.15


class OpenGLRenderer:
    def __init__(self, env: GridWorldEnv, config: RendererConfig | None = None) -> None:
        self.env = env
        self.config = config or RendererConfig()
        self.window: Any | None = None
        self.models: dict[str, Model3D] = {}

    def initialize(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(self.config.window_size, self.config.window_size, "Virtual World Explorer 3D", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create the OpenGL window")
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.config.window_size, self.config.window_size)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.08, 0.08, 0.1, 1.0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)

        light_ambient = [0.25, 0.25, 0.3, 1.0]
        light_diffuse = [0.9, 0.9, 0.95, 1.0]
        light_position = [5.0, 5.0, 15.0, 1.0]
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        model_paths = {
            "chair": "assets/models/chair/Chair.obj",
            "lamp": "assets/models/lamp/Standing_lamp_01.obj",
            "table": "assets/models/table/model.obj",
        }
        for label, path in model_paths.items():
            if os.path.exists(path):
                model = Model3D(path)
                model.load_textures_gl()
                self.models[label] = model
                v_count = sum(len(g['vertices']) for g in model.geometries)
                f_count = sum(len(g['faces']) for g in model.geometries)
                print(f"[Model3D] Caricato: {path} ({v_count} vertici, {f_count} facce)")
            else:
                print(f"[Model3D] WARNING: {path} non trovato")

    @staticmethod
    def _set_material(diffuse: tuple[float, float, float]) -> None:
        ambient = (diffuse[0] * 0.3, diffuse[1] * 0.3, diffuse[2] * 0.3)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [*diffuse, 1.0])

    def capture_frame(self) -> np.ndarray | None:
        if self.window is None:
            return None
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera(egocentric=True)
        self._draw_grid()
        self._draw_objects()
        self._draw_agent()
        
        width, height = self.config.window_size, self.config.window_size
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image_array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        image_array = np.flipud(image_array)
        return image_array

    def should_close(self) -> bool:
        return bool(self.window is None or glfw.window_should_close(self.window))

    def poll(self) -> None:
        if self.window is not None:
            glfw.poll_events()

    def draw(self) -> None:
        if self.window is None:
            return
            
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera(egocentric=False)
        self._draw_grid()
        self._draw_objects()
        self._draw_agent()
        
        self._draw_hud_overlay([
            f"TARGET: {self.env.target_label.upper()}",
            f"AGENT: {self.env.agent_x}, {self.env.agent_y}",
            f"VISIBLE: {self.env.detector.detect(self.env.objects, (self.env.agent_x, self.env.agent_y), self.env.target_label).visible}",
        ])
        glfw.swap_buffers(self.window)

    def _setup_camera(self, egocentric: bool) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        near_val = 0.1
        far_val = 50.0
        
        if egocentric:
            view_size = 3.5
            glOrtho(-view_size, view_size, -view_size, view_size, near_val, far_val)
        else:
            fov = 45.0
            top = near_val * math.tan(math.radians(fov / 2.0))
            right = top
            glFrustum(-right, right, -top, top, near_val, far_val)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        if egocentric:
            glTranslatef(-(self.env.agent_x + 0.5), -(self.env.agent_y + 0.5), -self.env.size * 1.5)
        else:
            glTranslatef(-self.env.size / 2.0, -self.env.size / 2.0, -self.env.size * 1.5)
            
        glRotatef(-55.0, 1.0, 0.0, 0.0)

    def shutdown(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def _draw_grid(self) -> None:
        glColor3f(0.18, 0.18, 0.22)
        glBegin(GL_LINES)
        for index in range(self.env.size + 1):
            glVertex3f(index, 0, 0)
            glVertex3f(index, self.env.size, 0)
            glVertex3f(0, index, 0)
            glVertex3f(self.env.size, index, 0)
        glEnd()

    def _draw_cube(self, x: float, y: float, size: float) -> None:
        z_bottom = 0.0
        z_top = size
        glBegin(GL_QUADS)
        # Superiore (normale +Z)
        glNormal3f(0, 0, 1)
        glVertex3f(x, y, z_top)
        glVertex3f(x + size, y, z_top)
        glVertex3f(x + size, y + size, z_top)
        glVertex3f(x, y + size, z_top)
        # Inferiore (normale -Z)
        glNormal3f(0, 0, -1)
        glVertex3f(x, y, z_bottom)
        glVertex3f(x, y + size, z_bottom)
        glVertex3f(x + size, y + size, z_bottom)
        glVertex3f(x + size, y, z_bottom)
        # Frontale (normale -Y)
        glNormal3f(0, -1, 0)
        glVertex3f(x, y, z_bottom)
        glVertex3f(x + size, y, z_bottom)
        glVertex3f(x + size, y, z_top)
        glVertex3f(x, y, z_top)
        # Destra (normale +X)
        glNormal3f(1, 0, 0)
        glVertex3f(x + size, y, z_bottom)
        glVertex3f(x + size, y + size, z_bottom)
        glVertex3f(x + size, y + size, z_top)
        glVertex3f(x + size, y, z_top)
        # Posteriore (normale +Y)
        glNormal3f(0, 1, 0)
        glVertex3f(x + size, y + size, z_bottom)
        glVertex3f(x, y + size, z_bottom)
        glVertex3f(x, y + size, z_top)
        glVertex3f(x + size, y + size, z_top)
        # Sinistra (normale -X)
        glNormal3f(-1, 0, 0)
        glVertex3f(x, y + size, z_bottom)
        glVertex3f(x, y, z_bottom)
        glVertex3f(x, y, z_top)
        glVertex3f(x, y + size, z_top)
        glEnd()

    def _draw_objects(self) -> None:
        model_scale = 0.55
        for scene_object in self.env.objects:
            model = self.models.get(scene_object.label)
            if model is None:
                continue

            glPushMatrix()
            glTranslatef(scene_object.x + 0.5, scene_object.y + 0.5, 0.0)

            mat_colors = [m.diffuse for m in model.materials.values()]
            if mat_colors:
                avg_color = (
                    sum(c[0] for c in mat_colors) / len(mat_colors),
                    sum(c[1] for c in mat_colors) / len(mat_colors),
                    sum(c[2] for c in mat_colors) / len(mat_colors),
                )
                self._set_material(avg_color)

            model.render(target_size=model_scale)
            glPopMatrix()

    def _draw_agent(self) -> None:
        padding = self.config.cell_padding + 0.04
        glDisable(GL_TEXTURE_2D)
        self._set_material((0.95, 0.95, 0.95))
        side = 1.0 - (padding * 2)
        self._draw_cube(self.env.agent_x + padding, self.env.agent_y + padding, side)

    def _draw_hud_overlay(self, lines: list[str]) -> None:
        if self.window is None:
            return
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.config.window_size, 0, self.config.window_size, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        start_x = 12.0
        start_y = self.config.window_size - 14.0
        panel_width = 165.0
        panel_height = 30.0
        glColor3f(0.05, 0.05, 0.07)
        glRectf(start_x - 6.0, start_y + 8.0, start_x + panel_width, start_y - panel_height)
        for index, line in enumerate(lines):
            self._draw_text(start_x + 6.0, start_y - index * self.config.hud_line_height, line, self.config.hud_scale)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_text(self, x: float, y: float, text: str, scale: float) -> None:
        cursor_x = x
        for character in text.upper():
            if character == " ":
                cursor_x += scale * 2.6
                continue
            glyph = _GLYPHS.get(character, _GLYPHS["?"])
            self._draw_glyph(cursor_x, y, glyph, scale)
            cursor_x += scale * 4.0

    def _draw_glyph(self, x: float, y: float, glyph: tuple[str, ...], scale: float) -> None:
        glColor3f(0.95, 0.95, 0.98)
        for row_index, row in enumerate(glyph):
            for column_index, pixel in enumerate(row):
                if pixel != "1":
                    continue
                left = x + column_index * scale
                top = y - row_index * scale
                glRectf(left, top - scale, left + scale, top)


_GLYPHS: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01110"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "01010", "01010", "00100"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ",": ("00000", "00000", "00000", "00000", "00100", "00100", "01000"),
    "0": ("01110", "10011", "10101", "10101", "10101", "11001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}