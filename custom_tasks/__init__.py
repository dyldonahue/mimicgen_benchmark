"""Custom tasks"""

from robosuite.environments.base import register_env

from .mug_tree import *
from .place_on_shelf import *
from .drawer_place import *
from .screw_touch import *

register_env(mug_tree)
register_env(screw_touch_d1)
register_env(screw_touch_d2)
register_env(drawer_place)
register_env(place_on_shelf)
