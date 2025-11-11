from robosuite.models.objects import MujocoXMLObject, CompositeBodyObject
import os
from robosuite.utils.transform_utils import axisangle2quat, quat_multiply
from . import get_object_xml_path
import numpy as np


class PegInHoleSquareObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = get_object_xml_path("peginhole-square")

        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class PegInHoleTriangleObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(
            parent_dir, "assets", "objects", "peginhole-triangle.xml"
        )

        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class PegInHoleBoardObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "peginhole-board.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class MugTreeObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "mugtree.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class MugObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "mug.xml")
        

        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class ShelfObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "shelf.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class PipeObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "pipe.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class ScrewDriverObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "screwdriver.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class ScrewObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "screw.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class ThreadedInsertObject(MujocoXMLObject):
    def __init__(self, name):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "threaded_insert.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class ScrewWithInsert(CompositeBodyObject):
    def __init__(self, name="screw_insert", screw_obj=None, insert_obj=None):
        assert (
            screw_obj is not None and insert_obj is not None
        ), "Must provide both screw and insert objects"

        # Build object list
        objects = [insert_obj, screw_obj]  # Insert is parent, screw is child

        # Positioning
        insert_pos = [0, 0, 0]
        insert_quat = [0, 0, 0, 1]

        screw_offset = [0.0, 0.0, 0.048]  # E.g., placed just above insert

        # Construct a 90° (π/2 rad) rotation around Y axis (can change axis as needed)
        axis = [0, 0, np.pi / 2]  # Y-axis
        screw_rel_quat = axisangle2quat(axis)

        # Final screw quaternion: relative to insert
        screw_quat = quat_multiply(screw_rel_quat, insert_quat)
        screw_pos = np.array(insert_pos) + np.array(screw_offset)

        # Parent-child relationship
        parents = [None, insert_obj.root_body]

        # Call super constructor
        super().__init__(
            name=name,
            objects=objects,
            object_locations=[insert_pos, screw_pos],
            object_quats=[insert_quat, screw_quat],
            object_parents=parents,
            joints=None,
        )

class CabinetObject(MujocoXMLObject):
    """
    Cabinet with three moveable drawers.
    
    Args:
        name (str): Name of this cabinet instance
    """
    
    def __init__(self, name):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "drawer.xml")
        
        super().__init__(
            xml_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class BookObject(MujocoXMLObject):
    def __init__(self, name):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        xml_path = os.path.join(parent_dir, "assets", "objects", "book.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
