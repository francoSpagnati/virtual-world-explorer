from __future__ import annotations

import math
import os
import numpy as np

from dataclasses import dataclass, field

from OpenGL.GL import (
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
import trimesh


@dataclass
class Material:
    name: str
    diffuse: tuple[float, float, float] = (0.5, 0.5, 0.5)
    ambient: tuple[float, float, float] = (0.2, 0.2, 0.2)
    texture_image: Image.Image | None = None
    texture_id: int | None = None


class Model3D:
    def __init__(self, obj_path: str) -> None:
        self.geometries: list[dict] = []
        self.materials: dict[str, Material] = {}
        
        scene = trimesh.load(obj_path)
        
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.items())
        else:
            geoms = [("default", scene)]
            
        all_vertices = []
        
        for name, geom in geoms:
            # Transform vertices: Y-up to Z-up
            v = geom.vertices
            v_transformed = np.column_stack((v[:, 0], v[:, 2], v[:, 1]))
            all_vertices.append(v_transformed)
            
            mat = Material(name=name)
            if hasattr(geom, 'visual') and geom.visual is not None:
                vis = geom.visual
                if hasattr(vis, 'material'):
                    m = vis.material
                    if hasattr(m, 'main_color'):
                        color = m.main_color[:3] / 255.0
                        mat.diffuse = tuple(color)
                    if hasattr(m, 'image') and m.image is not None:
                        mat.texture_image = m.image.copy().convert("RGBA")
                        mat.diffuse = (1.0, 1.0, 1.0)
            
            self.materials[name] = mat
            
            uvs = None
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                uvs = geom.visual.uv
                
            normals = geom.vertex_normals
            # Transform normals: Y-up to Z-up
            normals = np.column_stack((normals[:, 0], normals[:, 2], normals[:, 1]))
            
            self.geometries.append({
                'name': name,
                'vertices': v_transformed,
                'faces': geom.faces,
                'normals': normals,
                'uvs': uvs
            })
            
        if all_vertices:
            combined = np.vstack(all_vertices)
            min_bounds = combined.min(axis=0)
            max_bounds = combined.max(axis=0)
            self.center = (
                (min_bounds[0] + max_bounds[0]) / 2.0,
                (min_bounds[1] + max_bounds[1]) / 2.0,
                min_bounds[2]
            )
            dims = max_bounds - min_bounds
            max_dim = np.max(dims)
            self.normalized_scale = 1.0 / max_dim if max_dim > 0 else 1.0
        else:
            self.center = (0.0, 0.0, 0.0)
            self.normalized_scale = 1.0

    def load_textures_gl(self) -> None:
        for mat in self.materials.values():
            if mat.texture_image is not None:
                mat.texture_id = self._load_gl_texture(mat.texture_image)

    @staticmethod
    def _load_gl_texture(img: Image.Image) -> int:
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        try:
            img_data = img.tobytes("raw", "RGBA", 0, -1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        except Exception:
            fallback = bytes([200, 200, 200, 255])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, fallback)
        return tex_id

    def render(self, target_size: float = 0.7) -> None:
        glColor3f(1.0, 1.0, 1.0)

        for geom in self.geometries:
            mat = self.materials.get(geom['name'])
            has_texture = mat is not None and mat.texture_id is not None

            if has_texture:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, mat.texture_id)
            elif mat is not None:
                glColor3f(*mat.diffuse)

            glBegin(GL_TRIANGLES)
            vertices = geom['vertices']
            normals = geom['normals']
            uvs = geom['uvs']

            for face in geom['faces']:
                for v_idx in face:
                    x, y, z = vertices[v_idx]

                    x -= self.center[0]
                    y -= self.center[1]
                    z -= self.center[2]
                    x *= self.normalized_scale * target_size
                    y *= self.normalized_scale * target_size
                    z *= self.normalized_scale * target_size

                    nx, ny, nz = normals[v_idx]
                    glNormal3f(nx, ny, nz)

                    if has_texture and uvs is not None:
                        u, v = uvs[v_idx]
                        glTexCoord2f(u, v)

                    glVertex3f(x, y, z)
            glEnd()

            if has_texture:
                glDisable(GL_TEXTURE_2D)
