from custom_tasks.ScrewOnBolt import ScrewOnBoltTask
#from robosuite.robosuite.environments.manipulation.lift import Lift
from robosuite.controllers import load_controller_config
import numpy as np
from robosuite.utils.transform_utils import euler2mat, mat2quat, quat_multiply


#from robosuite.robots import Panda

controller_config = load_controller_config(default_controller="JOINT_POSITION")

env = ScrewOnBoltTask(
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    ignore_done=True,
    use_camera_obs=False,
    control_freq=20,
)

env.reset()


low, high = env.action_spec

for _ in range(1000):
    #action = np.random.uniform(low, high)
    #obs, reward, done, info = env.step(action)
    env.render()
