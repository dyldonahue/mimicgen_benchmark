import sys
from PIL import Image
import os

if len(sys.argv) < 2:
    print("Usage: python jpg_to_png.py <name>")
    sys.exit(1)

object_name = sys.argv[1]

model_dir = "/home/dyl/class/mimicgen_ws/assets/textures/"
model_file = os.path.join(model_dir, f"{object_name}.jpg")
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

os.chdir(model_dir)

# Derive output path by replacing extension
output_path = object_name + ".png"

# Open and convert
img = Image.open(model_file)
img.save(output_path, "PNG")

print(f"Saved: {output_path}")

if os.path.exists(model_file):
    os.remove(model_file)
    print(f"Deleted original file: {model_file}")
