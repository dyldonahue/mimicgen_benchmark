''' To use: '''
# pass in the object list to the config - ie ["pipeA", "pipeB"]

import mimicgen
from mimicgen.configs.config import MG_Config

class Stack_Config(MG_Config):
    """
    Corresponds to robosuite Stack task and variants.
    """
    NAME = "stack"
    TYPE = "robosuite"

    def __init__(self, object_list=None, **kwargs):
        super().__init__(**kwargs)
        self.object_list = object_list or ["pipeA", "pipeB"]


    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref=self.object_list[0], 
            subtask_term_signal="grasp",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref=self.object_list[1], 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()
