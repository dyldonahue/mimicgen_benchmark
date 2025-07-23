# my_collect_demos.py

from robosuite.scripts.collect_human_demonstrations import collect_human_trajectory
from custom_tasks.ScrewOnBolt import ScrewOnBolt

# Import and register custom components
from robosuite.environments.registry import register_env
register_env("ScrewOnBolt", ScrewOnBolt)

# Register custom objects or devices here as needed
# e.g., from custom_objects.my_block import MyBlock
# from robosuite.models.objects import register_custom_object
# register_custom_object("MyBlock", MyBlock)

if __name__ == "__main__":
    robosuite_main()