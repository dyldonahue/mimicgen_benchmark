from robosuite.devices.device import Device
from gello_software.experiments.run_env import main as gello_client_main, Args as gello_client_args
from gello_software.gello.zmq_core.robot_node import ZMQClientRobot
import numpy as np
import threading
import glob
from pathlib import Path
from gello_software.gello.data_utils.format_obs import save_frame
import time
from scipy.spatial.transform import Rotation as R
from robosuite.utils.transform_utils import quat2mat, mat2euler, quat_multiply, quat_inverse



# create Device class for Gello
class Gello(Device):
    """
    Gello device class.
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0, host="127.0.0.1", start_joints=None):
        super().__init__()

        # initialize the server, client, and the gello agent

        # client + server args are same 
        args = gello_client_args()
        args.agent = "gello"
        args.robot_type = "sim_ur"

        args.start_joints = start_joints if start_joints is not None else np.array([0, -1.57, 1.57, -1.57, -1.57, 0, 0])
        
        # launch the server in a separate thread
        #threading.Thread(target=launch_server, args=(args,)).start()

        # from gello_agent.py, were going to assume gello is first usb port found, unless specified otherwise
        from gello_software.gello.agents.gello_agent import GelloAgent
        if args.gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    args.gello_port = usb_ports[0]
                    print(f"using port {args.gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
        self.agent = GelloAgent(port=args.gello_port, start_joints=args.start_joints)
        self.joint_state = np.zeros(len(self.agent._robot._joint_ids))
        self.last_state = np.zeros(len(self.agent._robot._joint_ids))
        self.prev_ee_quat = None
        self.prev_ee_pos = None

        # Start polling
        self._running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

        # delay to ensure server is up
        import time
        time.sleep(2)
        

        # Connect to the server
        #self.client = ZMQClientRobot(port=args.robot_port)

        # test connection
        # try:
        #     self.client.num_dofs()
        # except Exception as e:
        #     print(f"Failed to connect to Gello server: {e}")
        #     raise

        # self.last_state = np.zeros(self.client.num_dofs())

        # self.last_state = np.zeros(self.client.num_dofs())
        # self.joint_state = self.last_state.copy()
        # self._joint_state = self.joint_state.copy()

        # self.thread = threading.Thread(target=self.run)
        # self.thread.daemon = True
        # self.thread.start()

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """

        # Initialize the robot
        pass

    def set_sim(self, sim, ee_site_name="gripper0_grip_site"):
      
        """
        Method to Set Mujuco simulation environment for the Gello device.
        This is needed to grab physical end effector position and orientation from mujuco apis
        """

        self.sim = sim
        
        self.model = sim.model   # Note underscore
        self.data = sim.data 
        self.ee_site_id = self.model.site_name2id(ee_site_name)
       




        
    def get_controller_state(self):
        """Returns the current state of the device, a dictionary of pos, orn, grasp, and reset."""

        # Read joint state from Gello
        self.joint_state = self.get_state()

        # need to patch a robosuite issue/inconvience - 
        # somewhere, the sim object must be getting wrapped and eliminating the model and data attributes
        # cannot find where this is happening, so we jsut manually set them here to the one we saved in set_sim
        self.sim.model = self.model
        self.sim.data = self.data
    
   
        self.data.qpos[:len(self.joint_state)] = self.joint_state
        
        self.sim.forward()

  
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        print (f"[GelloDevice] End Effector Position: {ee_pos}")
        from scipy.spatial.transform import Rotation as R

        rot_matrix = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        ee_quat = R.from_matrix(rot_matrix).as_quat()  
      

    
        if self.prev_ee_pos is None:
            dpos = np.zeros(3)
        else:
            dpos = ee_pos - self.prev_ee_pos

    
        if self.prev_ee_quat is None:
            raw_drotation = np.zeros(3)
            rotation = np.zeros(3)  
        else:
       
    
            delta_quat = quat_multiply(ee_quat, quat_inverse(self.prev_ee_quat))
            raw_drotation_mat = quat2mat(delta_quat)

            #important! patched numpy issue with mat2euler in source code (M = np.array--> np.asarray w/out copy)
            raw_drotation = mat2euler(raw_drotation_mat, axes='sxyz') 
            
            rotation =quat2mat(ee_quat)
            rotation = mat2euler(rotation, axes='sxyz')

    # Save current pose
        self.prev_ee_pos = ee_pos
        self.prev_ee_quat = ee_quat

        return dict(
            dpos=dpos,
            rotation=rotation *1,  
            raw_drotation=raw_drotation * 1,
            grasp=self.joint_state[-1],
            reset=False,
         )

        # dpos = self.joint_state[:3] - self.last_state[:3]  # Assuming first 3 joints are position
        # self.last_state = self.joint_state.copy()
    
        # return dict(
        #     dpos=dpos,  
        #     rotation=np.zeros(3),  
        #     raw_drotation=np.zeros(3),  
        #     grasp=self.joint_state[-1],
        #     reset=False
        # ) 
    
    def get_state(self):
    
        return self._joint_state




    def run(self):
        """
        Read the Gello state in a separate thread.
        """
        while self._running:
            try:
                self._joint_state = self.agent._robot.get_joint_state()
                #print(f"[GelloDevice] Joint state: {self._joint_state}")
            except Exception as e:
                print(f"[GelloDevice] Error reading joint state: {e}")
            
            # arbirary? can be tuned
            time.sleep(0.01)  
            

            
def launch_server(args):
    """
    Launch the Gello server.
    """
    if args.robot_type == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent/ "gello_software" / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from gello_software.gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=args.robot_port, host=args.hostname
    )
    

    server.serve()
    
   
    
if __name__ == "__main__":
    gello = Gello()
    
        


                   
                        

