from __future__ import annotations

from dataclasses import dataclass
import math
import os

import glfw
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,  # Per gestire la profondità 3D
    GL_DEPTH_TEST,        # Evita che gli oggetti dietro si sovrappongano davanti
    GL_LINES,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,             # Per disegnare le facce dei cubi e pannelli
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
    glVertex3f,           # Coordinate 3D (X, Y, Z)
    glViewport,
    glFrustum,            # Sostituisce gluPerspective in modo nativo
    glRotatef,            # Ruota la scena per inclinarla
    glTranslatef,         # Sposta la telecamera
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
    glReadPixels,         # Per catturare i pixel dello schermo
    GL_RGB,
)

# Usiamo PIL e NumPy per manipolare le immagini e i frame catturati
from PIL import Image
import numpy as np

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
        # Dizionario che conterrà i puntatori (ID) alle texture caricate in memoria grafica
        self.textures: dict[str, int] = {}

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
        
        # Abilitiamo il controllo della profondità hardware per il 3D
        glEnable(GL_DEPTH_TEST)
        
        # Abilitiamo la trasparenza (Alpha Blending) fondamentale per i billboard scontrornati
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glClearColor(0.08, 0.08, 0.1, 1.0)

        # Carichiamo le texture per i tre oggetti del mondo
        self.textures["chair"] = self._load_texture_from_file("assets/chair.png", (0.0, 1.0, 0.0)) # Backup verde
        self.textures["table"] = self._load_texture_from_file("assets/table.png", (0.0, 0.0, 1.0)) # Backup blu
        self.textures["lamp"] = self._load_texture_from_file("assets/lamp.png", (1.0, 0.0, 0.0))  # Backup rosso

    def _load_texture_from_file(self, filename: str, backup_color: tuple[float, float, float]) -> int:
        """Carica un'immagine da file e restituisce l'ID della texture OpenGL.
        Se il file non esiste, genera un pixel a tinta unita del colore di backup."""
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        
        if os.path.exists(filename):
            try:
                img = Image.open(filename).convert("RGBA")
                img_data = img.tobytes("raw", "RGBA", 0, -1)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
                print(f"[Texture] Caricata con successo l'immagine: {filename}")
                return texture_id
            except Exception as e:
                print(f"[Texture] Errore nel caricamento di {filename}: {e}. Uso il backup colorato.")
        
        r, g, b = [int(c * 255) for c in backup_color]
        fallback_data = bytes([r, g, b, 255])
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, fallback_data)
        return texture_id

    def capture_frame(self) -> np.ndarray:
        """Cattura lo schermo corrente renderizzato in memoria e lo restituisce come array NumPy RGB."""
        if self.window is None:
            return np.zeros((self.config.window_size, self.config.window_size, 3), dtype=np.uint8)
        
        width = self.config.window_size
        height = self.config.window_size
        
        # Leggiamo i pixel RGB direttamente dal buffer della scheda video
        glPixelStorei = 1 # Forza l'allineamento dei byte
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convertiamo i dati binari in un array NumPy ordinato (Altezza, Larghezza, Canali)
        image_array = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        
        # OpenGL renderizza dal basso verso l'alto (0,0 in basso a sinistra), 
        # le immagini standard vanno dall'alto in basso. Quindi ribaltiamo verticalmente l'array.
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
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        fov = 45.0
        near_val = 0.1
        far_val = 50.0
        top = near_val * math.tan(math.radians(fov / 2.0))
        right = top
        glFrustum(-right, right, -top, top, near_val, far_val)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glTranslatef(-self.env.size / 2.0, -self.env.size / 2.0, -self.env.size * 1.5)
        glRotatef(-55.0, 1.0, 0.0, 0.0)
        
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
            glVertex3f(index, 0, 0)
            glVertex3f(index, self.env.size, 0)
            glVertex3f(0, index, 0)
            glVertex3f(self.env.size, index, 0)
        glEnd()

    def _draw_cube(self, x: float, y: float, size: float) -> None:
        """Disegna un cubo solido 3D riga per riga sul piano (usato per l'agente)."""
        z_bottom = 0.0
        z_top = size
        glBegin(GL_QUADS)
        # Faccia superiore
        glVertex3f(x, y, z_top)
        glVertex3f(x + size, y, z_top)
        glVertex3f(x + size, y + size, z_top)
        glVertex3f(x, y + size, z_top)
        # Faccia inferiore
        glVertex3f(x, y, z_bottom)
        glVertex3f(x, y + size, z_bottom)
        glVertex3f(x + size, y + size, z_bottom)
        glVertex3f(x + size, y + size, z_bottom)
        # Faccia frontale
        glVertex3f(x, y, z_bottom)
        glVertex3f(x + size, y, z_bottom)
        glVertex3f(x + size, y, z_top)
        glVertex3f(x, y, z_top)
        # Faccia destra
        glVertex3f(x + size, y, z_bottom)
        glVertex3f(x + size, y + size, z_bottom)
        glVertex3f(x + size, y + size, z_top)
        glVertex3f(x + size, y, z_top)
        # Faccia posteriore
        glVertex3f(x + size, y + size, z_bottom)
        glVertex3f(x, y + size, z_bottom)
        glVertex3f(x, y + size, z_top)
        glVertex3f(x + size, y + size, z_top)
        # Faccia sinistra
        glVertex3f(x, y + size, z_bottom)
        glVertex3f(x, y, z_bottom)
        glVertex3f(x, y + size, z_top)
        glVertex3f(x, y + size, z_top)
        glEnd()

    def _draw_billboard(self, x: float, y: float, size: float, texture_id: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(x, y + size/2, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(x + size, y + size/2, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(x + size, y + size/2, size)
        glTexCoord2f(0.0, 1.0); glVertex3f(x, y + size/2, size)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def _draw_objects(self) -> None:
        padding = self.config.cell_padding
        for scene_object in self.env.objects:
            side = 1.0 - (padding * 2)
            tex_id = self.textures.get(scene_object.label, 0)
            self._draw_billboard(scene_object.x + padding, scene_object.y + padding, side, tex_id)

    def _draw_agent(self) -> None:
        padding = self.config.cell_padding + 0.04
        glColor3f(0.95, 0.95, 0.95)
        side = 1.0 - (padding * 2)
        self._draw_cube(self.env.agent_x + padding, self.env.agent_y + padding, side)

    def _draw_hud_overlay(self, lines: list[str]) -> None:
        if self.window is None:
            return
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
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