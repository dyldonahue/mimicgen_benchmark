"""Place On Shelf Task"""

#  Pick Up complex objects and place them on a shelf

import numpy as np
import os
import xml.etree.ElementTree as ET
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    UniformRandomSampler,
    SequentialCompositeSampler,
)
from robosuite.utils.transform_utils import convert_quat, rotation_matrix
from robosuite.utils.transform_utils import mat2quat, quat2mat
from robosuite.utils.sim_utils import check_contact


""" imported custom objects"""
from custom_objects.objects import ShelfObject
from custom_objects.objects import BookObject


class place_on_shelf(SingleArmEnv):
    """
    This class corresponds to the Place on Shelf task for a single robot arm.

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
        self.height_margin = 0.1

        # tolerances
        self._orientation_tolerance = 0.436 # radians off vertical, about 25 degrees

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
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

    def edit_model_xml(self, xml_str):
        curr_dir = os.path.dirname(__file__)
        assets_root = os.path.abspath(os.path.join(curr_dir, "../assets"))
        objects_path = os.path.join(assets_root, "objects")
        meshes_path = os.path.join(assets_root, "meshes")
        textures_path = os.path.join(assets_root, "textures")

        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        if asset is None:
            return xml_str

        all_elements = asset.findall("mesh") + asset.findall("texture")

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            if "robosuite" in old_path:

                continue

            filename = os.path.basename(old_path)

            if elem.tag == "mesh":
                new_path = os.path.join(meshes_path, filename)
            elif elem.tag == "texture":
                new_path = os.path.join(textures_path, filename)
            else:

                new_path = os.path.join(objects_path, filename)

            elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 4.0 is provided if the shelf tip touches the base of the screw

        Un-normalized components if using reward shaping:

            - Lifting: in [0, 1], to encourage the arm to reach the book anf lift it
            - Orienting: in [0, 1], non-zero if the book is upright
            - Locating: in [0, 1], non-zero if the book is within the shelf boundaries
            - Stable Contact: in [0, 1], non-zero if the book is stably on shelf for a few checks


        The reward is max over the following:

            - Lifting
            - Orienting
            - Locating
            - Contact

        The sparse reward only consists of the Contact component.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_lift, r_orient, r_location, r_contact = self.staged_rewards()
        if self.reward_shaping:
            reward = max(
                r_lift,
                r_orient,
                r_location,
                r_contact,
            )
        else:
            reward = 4.0 if r_contact > 0 else 0.0

        return reward

    def staged_rewards(self):
        r_lift = self._check_lift()
        r_orient = self._check_orientation()
        r_location = self._check_location()
        r_contact = self._check_stable_contact()

        return r_lift, r_orient, r_location, r_contact

    def _get_placement_initializer(self):

        self.placement_initializer = SequentialCompositeSampler(name="ObjectPlacement")

        # shelf is fixed in place
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="shelf",
                mujoco_objects=self.shelf,
                x_range=[-0.05, -0.05],
                y_range=[0.22, 0.22],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )
        )

        # book is placed randomly on the table
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="book",
                mujoco_objects=self.book,
                x_range=[
                    -self.table_full_size[0] / 2 + 0.2,
                    self.table_full_size[0] / 2 - 0.2,
                ],
                y_range=[
                    -self.table_full_size[1] / 2 + 0.2,
                    self.table_full_size[1] / 2 - 0.2,
                ],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
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
        self.shelf = ShelfObject(name="shelf")
        self.book = BookObject(name="book")

        # create a list of all objects to be placed
        objects = [self.shelf, self.book]

        # Create placement initializer
        self._get_placement_initializer()

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
        self.shelf_body_id = self.sim.model.body_name2id(self.shelf.root_body)
        self.book_body_id = self.sim.model.body_name2id(self.book.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.joints and len(obj.joints) > 0:
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

                else:
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat

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

            # position and rotation of the shelf
            @sensor(modality=modality)
            def shelf_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.shelf_body_id])

            @sensor(modality=modality)
            def shelf_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.shelf_body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def gripper_to_shelf(obs_cache):
                return (
                    obs_cache["shelf_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "shelf_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [shelf_pos, shelf_quat, gripper_to_shelf]
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
        Check if book is correctly sitting on the shelf.

        Returns:
            bool: True if book is sitting on the shelf, False otherwise.
        """
        _, _, _, r_contact = self.staged_rewards()

        return r_contact > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the shelf.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the shelf
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.shelf
            )

    def _check_lift(self):
        """
        Check if the book is lifted above the table top by a margin.

        Returns:
            bool: True if book is lifted, False otherwise.
        """

        if not self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.book
        ):
            return False

        z = self.sim.data.body_xpos[self.book_body_id][2]
        table_z = self.table_offset[2]

        lifted = z > table_z + self.height_margin

        if lifted:
            self._lifted += 1
            return self._lifted
        else:
            self._lifted = 0
            return False

    def _check_orientation(self):
        """
        Check if the book is oriented correctly to be placed on the shelf.

        Returns:
            bool: True if shelf is upright, False otherwise.
        """

        # get book quaternion
        quat = self.sim.data.body_xquat[self.book_body_id]
        rot_mat = quat2mat(quat)

        # bump axis 2 into index0 and take absolute val so that upright == identity matrix
        rot_mat = np.abs(rot_mat[[2, 0, 1], :])
        upright = np.allclose(rot_mat, np.eye(3), atol=self._orientation_tolerance)

        if upright:
            return True
        else:
            return False

    def _check_location(self):
        """
        Check if the book is within boundary of the right side of the shelf.

        Returns:
            bool: True if book is within shelf boundaries, False otherwise.
        """

        if self._check_orientation():

            # get book position
            book_pos = self.sim.data.body_xpos[self.book_body_id]

            # 0.15 is roughly the right quarter of the shelf length, and -0.05 is the left corner
            in_x = book_pos[0] > 0.15 and book_pos[0] < 0.3

            # shelf width starts at 0.35, it can hang off a bit
            in_y = book_pos[1] > 0.30 and book_pos[1] < 0.38

            # shelf z is at 0.92, but book position is at the center of mass
            # so we need to be above .92 + margin for book length
            in_z = book_pos[2] > 1.0


            #if book_pos[2] > 1.12:
               # print("height good")

            if in_x and in_y and in_z:
                return True
            else:
                return False

        else:
            return False

    def _check_stable_contact(self):
        """
        Check if the book is on shelf and does not fall off for a few consecutive checks.

        Returns:
            bool: True if book is stably in contact with shelf, False otherwise.
        """

        # location being correct implies it hasn't fallen off
        if self._check_location():

            if check_contact(
                self.sim, self.book.contact_geoms, self.shelf.contact_geoms
            ):

                self._stable_contact += 1

                if self._stable_contact >= 100:
                    return True
                else:
                    return False

            self._stable_contact = 0
            return False

        else:
            self._stable_contact = 0
            return False
