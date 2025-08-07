import os
import sys
import trimesh
import numpy as np
import xml.etree.ElementTree as ET

def generate_xml_from_stl(object_name, mesh_path):
    mesh = trimesh.load_mesh(mesh_path)

    bounds = mesh.bounds
    min_bounds = bounds[0]
    max_bounds = bounds[1]

    center = mesh.centroid
    bottom_site = [0, 0, float(min_bounds[2] - center[2])]
    top_site = [0, 0, float(max_bounds[2] - center[2])]
    horiz_radius_x = float((max_bounds[0] - min_bounds[0]) / 2)
    horiz_radius_y = float((max_bounds[1] - min_bounds[1]) / 2)

    scale = 0.0007
    bottom_site = [scale * (min_bounds[0] - center[0]),
               scale * (min_bounds[1] - center[1]),
               scale * (min_bounds[2] - center[2])]
    top_site = [scale * (max_bounds[0] - center[0]),
            scale * (max_bounds[1] - center[1]),
            scale * (max_bounds[2] - center[2])]
    horiz_radius_x = scale * ((max_bounds[0] - min_bounds[0]) / 2)
    horiz_radius_y = scale * ((max_bounds[1] - min_bounds[1]) / 2)


    mjcf = ET.Element("mujoco", model=object_name)
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", file=f"../meshes/{object_name}.stl", name=f"{object_name}_mesh", scale=f"{scale} {scale} {scale}")
    ET.SubElement(asset, "texture", file=f"../textures/{object_name}.png", type="2d", name=f"tex-{object_name}")
    ET.SubElement(asset, "material", name=object_name, reflectance="0.7", texrepeat="15 15", texture=f"tex-{object_name}", texuniform="true")

    worldbody = ET.SubElement(mjcf, "worldbody")
    parent_body = ET.SubElement(worldbody, "body")
    object_body = ET.SubElement(parent_body, "body", name="object")
    ET.SubElement(object_body, "geom",
                  pos="0 0 0", mesh=f"{object_name}_mesh", type="mesh",
                  solimp="0.998 0.998 0.001", solref="0.001 1",
                  density="50", friction="0.95 0.3 0.1",
                  material=object_name, group="0", condim="4")


    ET.SubElement(parent_body, "site", rgba="0 0 0 0", size="0.005",
                  pos=f"{bottom_site[0]} {bottom_site[1]} {bottom_site[2]}", name="bottom_site")
    ET.SubElement(parent_body, "site", rgba="0 0 0 0", size="0.005",
                  pos=f"{top_site[0]} {top_site[1]} {top_site[2]}", name="top_site")
    ET.SubElement(parent_body, "site", rgba="0 0 0 0", size="0.005",
                  pos=f"{horiz_radius_x} {horiz_radius_y} 0", name="horizontal_radius_site")

    import xml.dom.minidom
    rough_string = ET.tostring(mjcf, encoding="unicode")
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_xml.py <object_name>")
        sys.exit(1)

    name = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_path = os.path.join(script_dir, "..", "assets", "meshes", f"{name}.stl")
    output_dir = os.path.join(script_dir, "..", "assets", "objects")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.xml")

    if not os.path.isfile(mesh_path):
        print(f"Error: Mesh not found at {mesh_path}")
        sys.exit(1)

    xml_content = generate_xml_from_stl(name, mesh_path)

    with open(output_path, "w") as f:
        f.write(xml_content)

    print(f"Generated XML: {output_path}")
