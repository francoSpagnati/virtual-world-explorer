from __future__ import annotations

import math
import os
import re

from dataclasses import dataclass, field

from OpenGL.GL import (
    GL_QUADS,
    GL_TRIANGLES,
    glBegin,
    glEnd,
    glVertex3f,
    glNormal3f,
    glTexCoord2f,
    glColor3f,
    glEnable,
    glDisable,
    GL_TEXTURE_2D,
    glBindTexture,
    glGenTextures,
    glTexImage2D,
    glTexParameterf,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_LINEAR,
)
from PIL import Image


@dataclass
class Material:
    name: str
    diffuse: tuple[float, float, float] = (0.5, 0.5, 0.5)
    ambient: tuple[float, float, float] = (0.2, 0.2, 0.2)
    texture_path: str | None = None
    texture_id: int | None = None


@dataclass
class FaceGroup:
    material_name: str
    # Each face: list of (v_idx, vt_idx, vn_idx) using 0-based indexing
    faces: list[list[tuple[int, int, int]]] = field(default_factory=list)


class Model3D:
    def __init__(self, obj_path: str) -> None:
        self.obj_dir = os.path.dirname(os.path.abspath(obj_path))
        self.vertices: list[tuple[float, float, float]] = []
        self.texcoords: list[tuple[float, float]] = []
        self.normals: list[tuple[float, float, float]] = []
        self.materials: dict[str, Material] = {}
        self.face_groups: list[FaceGroup] = []

        self._parse_obj(obj_path)
        self._compute_bounds()

    def _parse_obj(self, path: str) -> None:
        current_group = FaceGroup(material_name="")
        mtl_file: str | None = None

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                keyword = parts[0]

                if keyword == "v":
                    self.vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif keyword == "vt":
                    self.texcoords.append((float(parts[1]), float(parts[2])))
                elif keyword == "vn":
                    self.normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif keyword == "f":
                    face: list[tuple[int, int, int]] = []
                    for token in parts[1:]:
                        indices = token.split("/")
                        v_idx = int(indices[0]) - 1 if indices[0] else -1
                        vt_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
                        vn_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else -1
                        face.append((v_idx, vt_idx, vn_idx))
                    current_group.faces.append(face)
                elif keyword == "usemtl":
                    if current_group.faces:
                        self.face_groups.append(current_group)
                    current_group = FaceGroup(material_name=parts[1])
                elif keyword == "mtllib":
                    mtl_file = " ".join(parts[1:])

        if current_group.faces:
            self.face_groups.append(current_group)

        if mtl_file:
            mtl_path = os.path.join(self.obj_dir, mtl_file)
            if os.path.exists(mtl_path):
                self._parse_mtl(mtl_path)

    def _parse_mtl(self, path: str) -> None:
        current_mat: Material | None = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                keyword = parts[0]

                if keyword == "newmtl":
                    if current_mat is not None:
                        self.materials[current_mat.name] = current_mat
                    current_mat = Material(name=parts[1])
                elif keyword == "Kd" and current_mat is not None:
                    current_mat.diffuse = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif keyword == "Ka" and current_mat is not None:
                    current_mat.ambient = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif keyword == "map_Kd" and current_mat is not None:
                    tex_name = " ".join(parts[1:])
                    tex_path = os.path.join(self.obj_dir, tex_name)
                    if os.path.exists(tex_path):
                        current_mat.texture_path = tex_path

        if current_mat is not None:
            self.materials[current_mat.name] = current_mat

    def load_textures_gl(self) -> None:
        for mat in self.materials.values():
            if mat.texture_path is not None:
                mat.texture_id = self._load_gl_texture(mat.texture_path)

    @staticmethod
    def _load_gl_texture(path: str) -> int:
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        try:
            img = Image.open(path).convert("RGBA")
            img_data = img.tobytes("raw", "RGBA", 0, -1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        except Exception:
            fallback = bytes([200, 200, 200, 255])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, fallback)
        return tex_id

    def _compute_bounds(self) -> None:
        if not self.vertices:
            self.min_bounds = self.max_bounds = (0.0, 0.0, 0.0)
            self.center = (0.0, 0.0, 0.0)
            self.normalized_scale = 1.0
            return

        # Transform OBJ Y-up to scene Z-up: (x, y, z) -> (x, z, y)
        transformed = [(v[0], v[2], v[1]) for v in self.vertices]

        xs = [v[0] for v in transformed]
        ys = [v[1] for v in transformed]
        zs = [v[2] for v in transformed]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        self.min_bounds = (min_x, min_y, min_z)
        self.max_bounds = (max_x, max_y, max_z)
        self.center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, min_z)  # XY center, Z at base

        dim_x = max_x - min_x
        dim_y = max_y - min_y
        dim_z = max_z - min_z
        max_dim = max(dim_x, dim_y, dim_z)
        self.normalized_scale = 1.0 / max_dim if max_dim > 0 else 1.0

    def _vertex(self, idx: int) -> tuple[float, float, float]:
        v = self.vertices[idx]
        return (v[0], v[2], v[1])

    def _normal(self, idx: int) -> tuple[float, float, float]:
        n = self.normals[idx]
        return (n[0], n[2], n[1])

    def render(self, target_size: float = 0.7) -> None:
        glColor3f(1.0, 1.0, 1.0)

        for group in self.face_groups:
            mat = self.materials.get(group.material_name)
            has_texture = mat is not None and mat.texture_id is not None

            if has_texture:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, mat.texture_id)
            elif mat is not None:
                glColor3f(*mat.diffuse)

            for face in group.faces:
                if len(face) < 3:
                    continue
                mode = GL_QUADS if len(face) == 4 else GL_TRIANGLES if len(face) == 3 else 0
                if mode == 0:
                    continue

                glBegin(mode)
                for v_idx, vt_idx, vn_idx in face:
                    x, y, z = self._vertex(v_idx)

                    x -= self.center[0]
                    y -= self.center[1]
                    z -= self.center[2]
                    x *= self.normalized_scale * target_size
                    y *= self.normalized_scale * target_size
                    z *= self.normalized_scale * target_size

                    if vn_idx >= 0 and vn_idx < len(self.normals):
                        nx, ny, nz = self._normal(vn_idx)
                        glNormal3f(nx, ny, nz)

                    if has_texture and vt_idx >= 0 and vt_idx < len(self.texcoords):
                        u, v = self.texcoords[vt_idx]
                        glTexCoord2f(u, v)

                    glVertex3f(x, y, z)
                glEnd()

            if has_texture:
                glDisable(GL_TEXTURE_2D)
