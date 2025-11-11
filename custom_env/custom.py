import mimicgen
from mimicgen.env_interfaces.robosuite import RobosuiteInterface
    
class MG_ScrewTouch(RobosuiteInterface):
    """
    Corresponds to robosuite Screwbolt task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        return dict(
            screwdriver=self.get_object_pose(obj_name=self.env.screwdriver.root_body, obj_type="body"),
            screw_with_insert=self.get_object_pose(obj_name=self.env.screwinsert.root_body, obj_type="body"),
        )
    
    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()
    
        signals["lift"] = int(self.env._check_lift())
        signals["orientation"] = int(self.env._check_orientation())
        signals["align"] = int(self.env._check_align())

        # final subtask is placing screwdriver on screw_with_insert (motion relative to screw_with_insert) - but final subtask signal is not needed
        return signals
    
class MG_PlaceOnShelf(RobosuiteInterface):
    """
    Corresponds to robosuite PlaceOnShelf task and variants.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stable_contact = 0

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        return dict(
            book=self.get_object_pose(obj_name=self.env.book.root_body, obj_type="body"),
            shelf=self.get_object_pose(obj_name=self.env.shelf.root_body, obj_type="body"),
        )
    
    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()
    
        signals["lift"] = int(self.env._check_lift())
        signals["orientation"] = int(self.env._check_orientation())
        signals["location"] = int(self.env._check_location())

        # final subtask is releasing book on shelf (motion relative to shelf) - but final subtask signal is not needed
        return signals


class MG_MugTree(RobosuiteInterface):
    """
    Corresponds to robosuite mug_tree task and variants.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stable_contact = 0

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        return dict(
            mug=self.get_object_pose(obj_name=self.env.mug.root_body, obj_type="body"),
            tree=self.get_object_pose(obj_name=self.env.tree.root_body, obj_type="body"),
        )
    
    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()
    
        signals["lift"] = int(self.env._check_lift())
        signals["orientation"] = int(self.env._check_orientation())

        # final subtask is releasing mug on tree - but final subtask signal is not needed
        return signals

class MG_Test(RobosuiteInterface):
    """
    Corresponds to robosuite mug_tree task and variants.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stable_contact = 0

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        return dict(
            mug=self.get_object_pose(obj_name=self.env.mug.root_body, obj_type="body"),
            tree=self.get_object_pose(obj_name=self.env.tree.root_body, obj_type="body"),
        )
    
    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()
    
        signals["lift"] = int(self.env._check_lift())
        signals["orientation"] = int(self.env._check_orientation())

        # final subtask is releasing mug on tree - but final subtask signal is not needed
        return signals
    

