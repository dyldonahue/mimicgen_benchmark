import mimicgen
from mimicgen.configs.config import MG_Config

class Screwbolt_Config(MG_Config):
    """
    Corresponds to robosuite Stack task and variants.
    """
    NAME = "screwbolt"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """

        # grasp & lift screwdriver
        self.task.task_spec.subtask_1 = dict(
            object_ref="screwdriver", 
            subtask_term_signal="lift",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # get close to contact
        # self.task.task_spec.subtask_2 = dict(
        #     object_ref="screw_with_insert", 
        #     subtask_term_signal="distance",
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )

        # touch screw with screwdriver
        self.task.task_spec.subtask_3 = dict(
            object_ref="screw_with_insert", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()
