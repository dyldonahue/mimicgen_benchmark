import sys

if len(sys.argv) < 2:
    print("Usage: python generate_xml.py <object_name>")
    sys.exit(1)

name = sys.argv[1]

xml_template = f'''<mujoco model="{name}">
  <asset>
    <mesh file="../meshes/{name}.stl" name="{name}_mesh" scale="0.003 0.003 0.003"/>
    <texture file="../textures/{name}.png" type="2d" name="tex-{name}" />
    <material name="{name}" reflectance="0.7" texrepeat="15 15" texture="tex-{name}" texuniform="true"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom pos="0 0 0" mesh="{name}_mesh" type="mesh" solimp="0.998 0.998 0.001" solref="0.001 1" density="50" friction="0.95 0.3 0.1" material="{name}" group="0" condim="4"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.045" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0.03" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0.03 0.03 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>'''

#place the XML in the assets/objects directory
import os
assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "objects")
output_file = os.path.join(assets_dir, f"{name}.xml")
with open(output_file, 'w') as f:
    f.write(xml_template)

print(f"Generated XML: {output_file}")
