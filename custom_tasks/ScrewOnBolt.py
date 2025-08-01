''' Scew On Bolt Task '''
# Pick up a screwdriver, and place into thread of a screw which is attached to a threaded insert and placed in any pose

import numpy as np
import random
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat, rotation_matrix
from robosuite.environments.base import register_env

from robosuite.utils.transform_utils import euler2mat, mat2quat, quat_multiply, quat2mat
from scipy.spatial.transform import Rotation as R



''' imported custom objects'''
from custom_objects.objects import ScrewDriverObject, ScrewWithInsert, ScrewObject, ThreadedInsertObject

class ScrewOnBoltTask(SingleArmEnv):
    """
    This class corresponds to the Screw On Bolt task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.insert_rot = None
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )


    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 4.0 is provided if the screwdriver tip touches the base of the screw

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the screwdriver and grasp it
            - Lifting: in {0, 0.5}, non-zero if arm has lifted the screwdriver above the table top by a margin
            - Aligning: in [0, 0.5], encourages moving screwdriver tip towards the screw thread
            - Touching: in {0, 2}, non-zero if the screwdriver tip is touching the screw head

        The reward is max over the following:

            - Reaching
            - Lifting
            - Aligning
            - Touching

        The sparse reward only consists of the touching component.

      
        Note that the final reward is normalized and scaled by
        reward_scale / 4.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_align, r_touch = self.staged_rewards()
        if self.reward_shaping:
            reward = max(
                r_reach,
                r_lift,
                r_align,
                r_touch,
            )
        else:
            reward = 4.0 if r_touch > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 4.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            4-tuple:

                - (float): reward for Reaching
                - (float): reward for Lifting
                - (float): reward for Aligning
                - (float): reward for Touching
        """
        # reaching is successful when the gripper site is close to the center handle of th screwdriver
        screwdriver_pos = self.sim.data.body_xpos[self.screwdriver_body_id]
        
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - screwdriver_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_screwdriver = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.screwdriver)
        if grasping_screwdriver:
            r_reach += 0.5
            

        # lifting is successful when the screwdriver is above the table top by a margin
        screwdriver_height = screwdriver_pos[2]
        table_height = self.table_offset[2]
        screwdriver_lifted = screwdriver_height > table_height + 0.8
        r_lift = 0.5 if screwdriver_lifted else 0.0

        
        # TODO need to test this alignment check
        # TODO is direction of the screwdriver best fit here? or is distance to the screw head better? Or combination of both?

        # Aligning is successful when screwdriver tip is facing the screw head
        if screwdriver_lifted:
            
            scewinsert_pos = self.sim.data.body_xpos[self.screwinsert_body_id]
            screwdriver_to_screwinsert = scewinsert_pos - screwdriver_pos
            screwdriver_to_screwinsert_norm = np.linalg.norm(screwdriver_to_screwinsert)
            screwdriver_to_screwinsert_unit = screwdriver_to_screwinsert / screwdriver_to_screwinsert_norm if screwdriver_to_screwinsert_norm > 0 else np.zeros(3)

            # forward along z?
            screwdriver_forward = self.sim.data.body_xmat[self.screwdriver_body_id][:3] 

            # calculate the angle between the screwdriver forward vector and the vector to the screw head
            angle_to_screw = np.arccos(np.clip(np.dot(screwdriver_forward, screwdriver_to_screwinsert_unit), -1.0, 1.0))

            # normalize the angle to be in [0, pi]
            angle_to_screw = np.abs(angle_to_screw)

            r_align = 0.5 * (1 - np.tanh(10.0 * angle_to_screw))  # reward is in [0, 0.5]
            
        else:
            r_align = 0.0
        

        # task is successful when the screwdriver tip is touching the screw head

        # top offsetis relative to the screw body position
        #self.screw.head = self.screw.top_offset + self.sim.data.body_xpos[self.cubeA_body_id]
        r_touch = 0

        #TODO: this may return true if any part of screwdriver is touching any part of the the screw, 
        # If so, adjust so only true if screwdriver tip is touching the screw head
        screwdriver_touching_screw = self.check_contact(self.screwdriver, self.screwinsert)   
        
        
        if grasping_screwdriver and screwdriver_touching_screw:
            r_touch = 2.0

        return r_reach, r_lift, r_align, r_touch
    

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
    
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        self.screwdriver = ScrewDriverObject(
            name="screwdriver"  
        )

        self.threaded_insert = ThreadedInsertObject(  
            name="threaded_insert"
        )

        self.screw = ScrewObject(
            name="screw")
        self.screwinsert = ScrewWithInsert(
            name="screw_with_insert",
            screw_obj=self.screw,
            insert_obj=self.threaded_insert

        )

        # create a list of all objects to be placed
        objects = [self.screwdriver, self.screwinsert]
        

        # Create placement initializer
        # if self.placement_initializer is not None:
            
        #     #self.placement_initializer.reset()
        #     #self.placement_initializer.add_objects(objects)
        # else:
            
        
        self.placement_initializer = SequentialCompositeSampler(name="ObjectPlacement")
        
        self.placement_initializer.append_sampler(
                 UniformRandomSampler(
                name="ScrewDriverSampler",
                mujoco_objects=[self.screwdriver],
                x_range=[-self.table_full_size[0]/2 + 0.1, self.table_full_size[0]/2 - 0.1],
                y_range=[-self.table_full_size[1]/2 + 0.1, self.table_full_size[1]/2 - 0.1],
                rotation_axis='z',
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

        self.placement_initializer.append_sampler(
                    UniformRandomSampler(
                    name="ScrewInsertSampler",
                    mujoco_objects=[self.screwinsert],
                    x_range=[-self.table_full_size[0]/2 + 0.1, self.table_full_size[0]/2 - 0.1],
                    y_range=[-self.table_full_size[1]/2 + 0.1, self.table_full_size[1]/2 - 0.1],
                    rotation_axis='z',
                    rotation=(np.pi, np.pi + np.pi/4),
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.04,
                ))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.screwdriver_body_id = self.sim.model.body_name2id(self.screwdriver.root_body)
        self.screwinsert_body_id = self.sim.model.body_name2id(self.screwinsert.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        print("Resetting Screw On Bolt Task")


        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            insert_pos, insert_quat, _ = object_placements["screw_with_insert"]

            # spin the direction of the tilted insert
            R_tilted = quat2mat(insert_quat)
            z_axis = np.array([0, 0, 1])
            angle = np.random.uniform(0, 2 * np.pi)
            R_spin = rotation_matrix(angle, z_axis)[:3, :3]
            R_final = R_spin @ R_tilted
            insert_quat = mat2quat(R_final)

            # gravity might not like tilt, freeze it so it doesnt tip over
            body_id = self.sim.model.body_name2id("screw_with_insert_threaded_insert_main")  # or whatever name it is
            self.sim.model.body_mass[body_id] = 0   
            self.sim.model.body_dofnum[body_id] = 0  # prevent movement
            self.sim.model.body_gravcomp[body_id] = 1.0
                             
            object_placements["screw_with_insert"] = (insert_pos, insert_quat, self.screwinsert)

            for obj_pos, obj_quat, obj in object_placements.values():

                if obj.joints and len(obj.joints) > 0:
                    # If the object has joints, set the joint positions
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                else:
                   try:
                        body_id = self.sim.model.body_name2id(obj.root_body)
                   except Exception:
                     raise ValueError(f"Could not find body name: {obj.root_body} in the model")

                   self.sim.model.body_pos[body_id] = obj_pos
                   self.sim.model.body_quat[body_id] = convert_quat(obj_quat, to="wxyz")
               

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the screwdriver
            @sensor(modality=modality)
            def screwdriver_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.screwdriver_body_id])

            @sensor(modality=modality)
            def screwdriver_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.screwdriver_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def screwinsert_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.screwinsert_body_id])
            
            @sensor(modality=modality)
            def screwinsert_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.screwinsert_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_screwdriver(obs_cache):
                return (
                    obs_cache["screwdriver_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "screwdriver_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_screwinsert(obs_cache):
                return (
                    obs_cache["screwinsert_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "screwinsert_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def screwdriver_to_screwinsert(obs_cache):
                return (
                    obs_cache["screwinsert_pos"] - obs_cache["screwdriver_pos"]
                    if "screwinsert_pos" in obs_cache and "screwdriver_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [screwdriver_pos, screwdriver_quat, 
                       screwinsert_pos, screwinsert_quat, 
                       gripper_to_screwdriver, gripper_to_screwinsert, screwdriver_to_screwinsert]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _check_success(self):
        """
        Check if screwdriver is correctly touching the screw head.

        Returns:
            bool: True if screwdriver is touching the screw head, False otherwise.
        """
        _, _, _, r_touch = self.staged_rewards()

        if (r_touch > 0):
            print("TASK COMPLETION IN PROGRESS")
        return r_touch > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the screwdriver.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the screwdriver
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.screwdriver)