import mimicgen
from mimicgen.env_interfaces.robosuite import RobosuiteInterface

class MG_Stack(RobosuiteInterface):
    """
    Corresponds to robosuite Stack task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        return dict(
            pipeA=self.get_object_pose(obj_name=self.env.pipeA.root_body, obj_type="body"),
            pipeB=self.get_object_pose(obj_name=self.env.pipeB.root_body, obj_type="body"),
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

        # first subtask is grasping pipeA (motion relative to pipeA)
        signals["grasp"] = int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.pipeA))

        # final subtask is placing pipeA on pipeB (motion relative to pipeB) - but final subtask signal is not needed
        return signals
