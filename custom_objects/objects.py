from robosuite.models.objects import MujocoXMLObject
import os
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion

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
            joints=[dict(type="free", damping="0.0005")],
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
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )






