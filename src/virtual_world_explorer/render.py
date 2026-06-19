from __future__ import annotations

from dataclasses import dataclass

import glfw
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_LINES,
    GL_MODELVIEW,
    GL_PROJECTION,
    glDisable,
    glEnable,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glRectf,
    glVertex2f,
    glViewport,
)

from .env import GridWorldEnv


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
        self.window = None

    def initialize(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(self.config.window_size, self.config.window_size, "Virtual World Explorer", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create the OpenGL window")
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.config.window_size, self.config.window_size)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.env.size, 0, self.env.size, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClearColor(0.08, 0.08, 0.1, 1.0)

    def should_close(self) -> bool:
        return bool(self.window is None or glfw.window_should_close(self.window))

    def poll(self) -> None:
        if self.window is not None:
            glfw.poll_events()

    def draw(self) -> None:
        if self.window is None:
            return
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        self._draw_grid()
        self._draw_objects()
        self._draw_agent()
        self._draw_hud_overlay([
            f"TARGET: {self.env.target_label.upper()}",
            f"AGENT: {self.env.agent_x}, {self.env.agent_y}",
            f"VISIBLE: {self.env.detector.detect(self.env.objects, (self.env.agent_x, self.env.agent_y), self.env.target_label).visible}",
        ])
        glfw.swap_buffers(self.window)

    def shutdown(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def _draw_grid(self) -> None:
        glColor3f(0.18, 0.18, 0.22)
        glBegin(GL_LINES)
        for index in range(self.env.size + 1):
            glVertex2f(index, 0)
            glVertex2f(index, self.env.size)
            glVertex2f(0, index)
            glVertex2f(self.env.size, index)
        glEnd()

    def _draw_objects(self) -> None:
        padding = self.config.cell_padding
        for scene_object in self.env.objects:
            glColor3f(*scene_object.color)
            glRectf(scene_object.x + padding, scene_object.y + padding, scene_object.x + 1 - padding, scene_object.y + 1 - padding)

    def _draw_agent(self) -> None:
        padding = self.config.cell_padding + 0.04
        glColor3f(0.95, 0.95, 0.95)
        glRectf(self.env.agent_x + padding, self.env.agent_y + padding, self.env.agent_x + 1 - padding, self.env.agent_y + 1 - padding)

    def _draw_hud_overlay(self, lines: list[str]) -> None:
        if self.window is None:
            return
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

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.env.size, 0, self.env.size, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

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

