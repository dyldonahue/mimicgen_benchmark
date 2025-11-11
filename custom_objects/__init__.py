"""Custom objects."""
from pathlib import Path

ASSETS_ROOT = Path(__file__).parent.parent / "assets"

def get_object_xml_path(obj_name):
    return str(ASSETS_ROOT / "objects" f"{obj_name}.xml")

from .objects import *