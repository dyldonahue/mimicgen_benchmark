
import os
import mujoco
import mujoco.viewer
import time
import sys

if len(sys.argv) < 2:
    print("Usage: python test_object.py <object_name>")
    sys.exit(1)

object_name = sys.argv[1]

model_dir = "/home/dyl/class/mimicgen_ws/assets/objects/"
model_file = os.path.join(model_dir, f"{object_name}.xml")
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

os.chdir(model_dir)

with open(model_file) as f:
    xml_string = f.read()

mj_model = mujoco.MjModel.from_xml_string(xml_string)
mj_data = mujoco.MjData(mj_model)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        time.sleep(1.0 / 60.0)
