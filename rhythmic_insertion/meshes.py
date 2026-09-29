"""Collision geometry for the bundled wrench and nut URDFs."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def load_collision_mesh(urdf_path):
    """Load the first link's untransformed mesh, as used by the task's SDFs."""
    urdf_path = Path(urdf_path)
    collisions = ET.parse(urdf_path).getroot().find('link').findall('collision')
    if len(collisions) != 1 or collisions[0].find('origin') is not None:
        raise ValueError('Expected one collision mesh at the link origin.')
    mesh_node = collisions[0].find('geometry/mesh')
    if mesh_node is None:
        raise ValueError('Expected mesh collision geometry.')
    scale = np.fromstring(mesh_node.get('scale', '1 1 1'), sep=' ')
    if not np.array_equal(scale, np.ones(3)):
        raise ValueError('Bundled collision meshes must have unit scale.')
    loaded = trimesh.load(str(urdf_path.parent / mesh_node.attrib['filename']))
    if isinstance(loaded, trimesh.Scene):
        parts = list(loaded.dump())
        return trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError('Expected a triangle mesh.')
    return loaded
