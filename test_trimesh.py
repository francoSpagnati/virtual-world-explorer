import trimesh
import numpy as np
from dataclasses import dataclass
from PIL import Image

@dataclass
class Material:
    diffuse: tuple[float, float, float] = (0.5, 0.5, 0.5)
    texture_image: Image.Image | None = None
    texture_id: int | None = None

class Model3D:
    def __init__(self, obj_path: str):
        self.geometries = []
        self.materials = []
        
        scene = trimesh.load(obj_path, force='mesh') 
        # force='mesh' might not work to flatten if there are multiple materials. Let's see.
        
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
        else:
            geoms = [scene]
            
        all_vertices = []
        
        for geom in geoms:
            # Transform vertices: Y-up to Z-up
            # x, y, z -> x, z, y
            v = geom.vertices
            v_transformed = np.column_stack((v[:, 0], v[:, 2], v[:, 1]))
            all_vertices.append(v_transformed)
            
            mat = Material()
            if hasattr(geom, 'visual') and geom.visual is not None:
                vis = geom.visual
                if hasattr(vis, 'material'):
                    m = vis.material
                    # simple material
                    if hasattr(m, 'main_color'):
                        color = m.main_color[:3] / 255.0
                        mat.diffuse = tuple(color)
                    if hasattr(m, 'image') and m.image is not None:
                        mat.texture_image = m.image
            
            # UVs
            uvs = None
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                uvs = geom.visual.uv
                
            # Normals
            # If vertex_normals doesn't exist or is empty, we can generate them or use face normals
            normals = geom.vertex_normals
            # transform normals to Z-up
            normals = np.column_stack((normals[:, 0], normals[:, 2], normals[:, 1]))
            
            self.geometries.append({
                'vertices': v_transformed,
                'faces': geom.faces,
                'normals': normals,
                'uvs': uvs
            })
            self.materials.append(mat)
            
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

model = Model3D("assets/models/chair/Chair.obj")
print(model.center)
print(model.normalized_scale)
print(len(model.geometries))
print(model.materials[0].texture_image)
