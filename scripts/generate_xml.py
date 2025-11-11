#!/usr/bin/env python3
"""
Complete STL object processor for RoboSuite.

Processes STL files for use in RoboSuite by:
- Analyzing geometry and printing stats
- Scaling to appropriate table size
- Simplifying for collision detection
- Generating RoboSuite-compatible XML
- Creating both visual and collision meshes

Usage: python generate_xml.py <object_name> [options]
Example: python process_stl_object.py mug --scale 0.001 --target-size 0.08
"""

import sys
import os
import argparse
import numpy as np
import trimesh
from xml.dom import minidom
import xml.etree.ElementTree as ET
import pymeshlab as ml


def analyze_mesh(mesh, name="object"):
    """
    Analyze mesh and print comprehensive statistics.

    Args:
        mesh: trimesh.Trimesh object
        name: Name of the object
    """

    # Basic geometry stats
    print(f"  Vertices:          {len(mesh.vertices):,}")
    print(f"  Faces:             {len(mesh.faces):,}")
    print(f"  Edges:             {len(mesh.edges):,}")

    # Topology
    print("\n")
    print(f"  Watertight:        {'✓' if mesh.is_watertight else '✗'}")
    print(f"  Convex:            {'✓' if mesh.is_convex else '✗'}")
    print(f"  Volume:            {mesh.volume:.6f} m³")
    print(f"  Surface area:      {mesh.area:.6f} m²")

    # Dimensions
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    print("\n")
    print(
        f"  Size (X×Y×Z):      {dimensions[0]:.4f} x {dimensions[1]:.4f} × {dimensions[2]:.4f} m"
    )
    print(
        f"  Bounding box min:  [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}, {bounds[0][2]:.4f}]"
    )
    print(
        f"  Bounding box max:  [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}, {bounds[1][2]:.4f}]"
    )
    print(
        f"  Center of mass:    [{mesh.center_mass[0]:.4f}, {mesh.center_mass[1]:.4f}, {mesh.center_mass[2]:.4f}]"
    )
    print(f"  Max dimension:     {max(dimensions):.4f} m")

    # Quality metrics
    print("\n")

    # Check for degenerate faces
    face_areas = mesh.area_faces
    degenerate = np.sum(face_areas < 1e-10)
    print(f"  Degenerate faces:  {degenerate}")

    # Check for duplicate vertices
    unique_verts = len(np.unique(mesh.vertices.view([("", mesh.vertices.dtype)] * 3)))
    duplicate_verts = len(mesh.vertices) - unique_verts
    print(f"  Duplicate verts:   {duplicate_verts}")

    # Aspect ratio
    aspect_ratio = max(dimensions) / (min(dimensions) + 1e-10)
    print(f"  Aspect ratio:      {aspect_ratio:.2f}")

    # Mesh density
    density = len(mesh.vertices) / (mesh.area + 1e-10)
    print(f"  Vertex density:    {density:.2f} verts/m²")

    # Recommendations - only if this is on the before editing call
    if name.endswith("(processed)"):
        return
    print("\n Based on the info above:")
    if not mesh.is_watertight:
        print(" Mesh has holes - will attempt to repair")
    if not mesh.is_winding_consistent:
        print("  Inconsistent face normals - will attempt to fix")
    if len(mesh.vertices) > 5000:
        print("  High vertex count - will create simplified collision mesh")
    if degenerate > 0:
        print("  Has degenerate faces - will clean")
    if duplicate_verts > 0:
        print("  Has duplicate vertices - will merge")
    if not mesh.is_convex:
        print(
            "  Mesh is concave - consider convex decomposition for accurate collision"
        )

    # Size recommendation for RoboSuite
    max_dim = max(dimensions)
    print("\nSize Guide:")
    if max_dim < 0.03:
        print(f"  Very small object ({max_dim*1000:.1f}mm) - good for small items")
    elif max_dim < 0.10:
        print(f"  Good size for tabletop manipulation ({max_dim*1000:.1f}mm)")
    elif max_dim < 0.30:
        print(
            f"  Medium-large object ({max_dim*1000:.1f}mm) - may need larger workspace"
        )
    else:
        print(f" Very large object ({max_dim*1000:.1f}mm) - consider scaling down")

    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "is_convex": mesh.is_convex,
        "dimensions": dimensions,
        "max_dimension": max(dimensions),
        "volume": mesh.volume,
        "needs_simplification": len(mesh.vertices) > 5000,
        "needs_repair": not mesh.is_watertight or degenerate > 0,
    }


def scale_mesh(mesh, target_scale=None, target_size=None):
    """
    Scale mesh to appropriate RoboSuite size.

    Args:
        mesh: trimesh.Trimesh object
        target_scale: Direct scale factor (e.g., 0.001 for mm->m)
        target_size: Target max dimension in meters (e.g., 0.08 for 8cm)

    Returns:
        Scaled mesh and scale factor used
    """
    original_dims = mesh.bounds[1] - mesh.bounds[0]
    original_max = max(original_dims)

    if target_scale is not None:
        scale_factor = target_scale
        print(f"\nScale factor provided: {scale_factor:.6f}")
    elif target_size is not None:
        scale_factor = target_size / original_max
        print(
            f"\nCalculated scale factor: {scale_factor:.6f} (to achieve {target_size}m max)"
        )

    if scale_factor != 1.0:
        mesh.apply_scale(scale_factor)
        new_dims = mesh.bounds[1] - mesh.bounds[0]
        new_max = max(new_dims)

    return mesh, scale_factor


def clean_mesh(mesh):
    """
    Clean and repair mesh.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        Cleaned mesh
    """
    original_verts = len(mesh.vertices)
    original_faces = len(mesh.faces)

    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    # Fill holes if not watertight
    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
            if not mesh.is_watertight:
                print("Note - still not watertight, could not fully repair holes")
        except Exception as e:
            print(f"Could not fill holes: {e}")

    # Fix normals
    if not mesh.is_winding_consistent:
        mesh.fix_normals()

    return mesh


def simplify_mesh(mesh, target_faces=1000, method="quadric"):
    """
    Simplify mesh for collision detection.

    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target number of faces
        method: 'quadric' (better quality) or 'fast'

    Returns:
        Simplified mesh
    """

    original_verts = len(mesh.vertices)
    original_faces = len(mesh.faces)

    print(f"\nOriginal: {original_verts:,} vertices, {original_faces:,} faces")
    print(f"Target: ~{target_faces:,} faces")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        mesh.export(tmp.name)
        tmp_path = tmp.name

    ms = ml.MeshSet()
    ms.load_new_mesh(tmp_path)

    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # Export and reload
    ms.save_current_mesh(tmp_path)
    simplified = trimesh.load(tmp_path)

    os.unlink(tmp_path)

    new_verts = len(simplified.vertices)
    new_faces = len(simplified.faces)

    reduction_verts = (1 - new_verts / original_verts) * 100
    reduction_faces = (1 - new_faces / original_faces) * 100

    print(f"\nSimplified Result: {new_verts:,} vertices, {new_faces:,} faces")
    print(f"Reduction: {reduction_verts:.1f}% vertices, {reduction_faces:.1f}% faces")

    return simplified


def center_mesh(mesh):
    """
    Center mesh so it sits on the table (z_min = 0) and is centered in x,y.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        Centered mesh and offset applied
    """

    bounds = mesh.bounds

    # Center in X and Y
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2

    # Bottom at Z=0
    z_min = bounds[0][2]

    offset = np.array([center_xy[0], center_xy[1], z_min])

    print(f"\nOriginal position:")
    print(f"  Center (XY): [{center_xy[0]:.4f}, {center_xy[1]:.4f}]")
    print(f"  Bottom (Z):  {z_min:.4f}")

    mesh.apply_translation(-offset)

    new_bounds = mesh.bounds
    print(f"\nCentered position:")
    print(f"  Center (XY): [0.0000, 0.0000]")
    print(f"  Bottom (Z):  {new_bounds[0][2]:.4f}")

    return mesh, offset

def generate_xml(
    output_path,
    visual_mesh_file,
    collision_mesh_file,
    object_name,
    rgba=None,
    density=500.0,
    use_same_for_collision=False,
):
    """
    Generate RoboSuite-compatible XML.

    Args:
        output_path: Where to save XML
        visual_mesh_file: Path to visual STL
        collision_mesh_file: Path to collision STL (or None to use visual)
        object_name: Name of the object
        rgba: Visual color [r, g, b, a]
        density: Density for physics
        use_same_for_collision: If True, use visual mesh for collision too
    """

    if rgba is None:
        rgba = [0.8, 0.8, 0.8, 1.0]

    xml = ET.Element("mujoco", model=object_name)

    # Asset section
    asset = ET.SubElement(xml, "asset")
    ET.SubElement(asset, "mesh", name="visual_mesh", file=os.path.join(".", "..", "meshes", visual_mesh_file))

    if not use_same_for_collision and collision_mesh_file is not None:
        ET.SubElement(asset, "mesh", name="collision_mesh", file=os.path.join(".", "..", "meshes", collision_mesh_file))

    # Worldbody
    worldbody = ET.SubElement(xml, "worldbody")
    body_outer = ET.SubElement(worldbody, "body")
    body = ET.SubElement(body_outer, "body", name="object")

    # Visual geom
    ET.SubElement(
        body,
        "geom",
        name="visual",
        type="mesh",
        mesh="visual_mesh",
        rgba=" ".join(map(str, rgba)),
        contype="0",
        conaffinity="0",
        group="1",
        density="1",
    )

    # Collision geom
    if use_same_for_collision:
        collision_mesh = "visual_mesh"
    else:
        collision_mesh = "collision_mesh"

    ET.SubElement(
        body,
        "geom",
        name="collision",
        type="mesh",
        mesh=collision_mesh,
        rgba="0.5 0.5 0.5 0.3",
        density=str(density),
        friction="0.95 0.3 0.1",
        condim="4",
        solref="0.001 1",
        solimp="0.998 0.998 0.001",
        group="0",
    )

    # Sites
    ET.SubElement(
        body_outer,
        "site",
        rgba="0 0 0 0",
        size="0.005",
        pos="0 0 0",
        name="bottom_site",
    )
    ET.SubElement(
        body_outer, "site", rgba="0 0 0 0", size="0.005", pos="0 0 0.1", name="top_site"
    )
    ET.SubElement(
        body_outer,
        "site",
        rgba="0 0 0 0",
        size="0.005",
        pos="0.05 0 0.05",
        name="horizontal_radius_site",
    )

    # Pretty print
    xml_str = ET.tostring(xml, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    lines = [line for line in pretty_xml.split("\n") if line.strip()]
    pretty_xml = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(pretty_xml)

    print(f"Generated XML: {output_path}")
 
def process_stl_object(
    input_path,
    object_name,
    output_dir=None,
    scale=None,
    target_size=None,
    collision_faces=1000,
    skip_simplification=False,
):
    """
    Complete pipeline to process STL for RoboSuite.

    Args:
        input_path: Path to input STL file
        object_name: Name of the object
        output_dir: Output directory (default: auto)
        scale: Scale factor (e.g., 0.001)
        target_size: Target max dimension in meters
        collision_faces: Target faces for collision mesh
        skip_simplification: Don't create separate collision mesh
    """

    # Setup output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_output_dir = os.path.join(script_dir, "..", "assets", "objects")
        mesh_output_dir = os.path.join(script_dir, "..", "assets", "meshes")

    try:
        mesh = trimesh.load(input_path)
    except Exception as e:

        print(f"Failed to load mesh: {e}")
        return False

    # Analyze original
    print("=" * 70)
    print(f"Before Editing: {object_name}")
    print("=" * 70)
    stats = analyze_mesh(mesh, object_name)

    # Clean mesh
    mesh = clean_mesh(mesh)

    # Scale mesh
    mesh, scale_factor = scale_mesh(mesh, target_scale=scale, target_size=target_size)

    # Center mesh
    mesh, offset = center_mesh(mesh)

    print("=" * 70)
    print("After Editing:")
    print("=" * 70)
    analyze_mesh(mesh, f"{object_name} (processed)")

    # Save visual mesh
    visual_path = os.path.join(mesh_output_dir, f"{object_name}_visual.stl")
    mesh.export(visual_path)

    # Create collision mesh
    if skip_simplification or stats["vertices"] < collision_faces * 3:
        collision_path = None
        use_same_for_collision = True
    else:
        collision_mesh = simplify_mesh(mesh, target_faces=collision_faces)
        collision_path = os.path.join(mesh_output_dir, f"{object_name}_collision.stl")
        collision_mesh.export(collision_path)
        use_same_for_collision = False

    # Generate XML
    xml_path = os.path.join(xml_output_dir, f"{object_name}.xml")
    generate_xml(
        output_path=xml_path,
        visual_mesh_file=f"{object_name}_visual.stl",
        collision_mesh_file=f"{object_name}_collision.stl" if collision_path else None,
        object_name=object_name,
        use_same_for_collision=use_same_for_collision,
    )

    print(f"\nGenerated files:")
    print(f"  - {object_name}.xml")
    print(f"  - {object_name}_visual.stl")
    if not use_same_for_collision:
        print(f"  - {object_name}_collision.stl")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process STL files for RoboSuite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect scale)
  python process_stl_object.py mug
  
  # Specify scale factor (e.g., mm to m)
  python process_stl_object.py mug --scale 0.001
  
  # Specify target size
  python process_stl_object.py mug --target-size 0.08
  
  # High detail collision
  python process_stl_object.py mug --collision-faces 2000
  
  # Use same mesh for collision (no simplification)
  python process_stl_object.py simple_box --skip-simplification

Expected file structure:
  assets/meshes/mug.stl  (input)
  assets/objects/mug/    (output)
        """,
    )

    parser.add_argument("object_name", help="Name of the object (e.g., 'mug')")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Scale factor (e.g., 0.001 for mm->m)",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        default=0.08,
        help="Target max dimension in meters (e.g., 0.08)",
    )
    parser.add_argument(
        "--collision-faces",
        type=int,
        default=1000,
        help="Target faces for collision mesh (default: 1000)",
    )
    parser.add_argument(
        "--skip-simplification",
        action="store_true",
        help="Use visual mesh for collision (no separate collision mesh)",
    )
    parser.add_argument("--output-dir", help="Output directory (default: auto)")

    args = parser.parse_args()

    # Find input mesh
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_path = os.path.join(
        script_dir, "..", "assets", "meshes", f"{args.object_name}.stl"
    )

    if not os.path.isfile(mesh_path):
        print(f"Error: Mesh not found at {mesh_path}")
        print(f"\nExpected location: assets/meshes/{args.object_name}.stl")
        sys.exit(1)

    # Process
    success = process_stl_object(
        input_path=mesh_path,
        object_name=args.object_name,
        output_dir=args.output_dir,
        scale=args.scale,
        target_size=args.target_size,
        collision_faces=args.collision_faces,
        skip_simplification=args.skip_simplification,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
