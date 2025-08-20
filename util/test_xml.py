import os
import sys
import trimesh
import numpy as np

def print_mesh_extents(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    
    # vertices in local coordinates
    verts = mesh.vertices.copy()
    
    # compute offsets along each axis
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    
    # offsets from origin
    offset_x = (min_xyz[0], max_xyz[0])
    offset_y = (min_xyz[1], max_xyz[1])
    offset_z = (min_xyz[2], max_xyz[2])

    # apply scale
    scale = 0.002
    offset_x = (offset_x[0] * scale, offset_x[1] *scale)
    offset_y = (offset_y[0] * scale, offset_y[1] *scale)
    offset_z = (offset_z[0] * scale, offset_z[1] * scale)

    # assume origin is at center, offset is half the full range
    
    print(f"Mesh: {mesh_path}")
    print(f"X offsets: min {offset_x[0]:.6f}, max {offset_x[1]:.6f}")
    print(f"Y offsets: min {offset_y[0]:.6f}, max {offset_y[1]:.6f}")
    print(f"Z offsets: min {offset_z[0]:.6f}, max {offset_z[1]:.6f}")
    print(f"Width X: {(offset_x[1] - offset_x[0])/2:.6f}")
    print(f"Width Y: {(offset_y[1] - offset_y[0]/2):.6f}")
    print(f"Height Z: {(offset_z[1] - offset_z[0])/2:.6f}")

if __name__ == "__main__":
    name = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_path = os.path.join(script_dir, "..", "assets", "meshes", f"{name}.stl")
    output_dir = os.path.join(script_dir, "..", "assets", "objects")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.xml")

    if not os.path.isfile(mesh_path):
        print(f"Error: Mesh not found at {mesh_path}")
        sys.exit(1)


    
    print_mesh_extents(mesh_path)
