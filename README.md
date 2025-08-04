# Contents
- [Environment Setup](#env-setup)
- [Directory Structure](#dir-struct)
- [Generating Data](#data-gen)

 <a id="env-setup"></a>
### Environment Setup:

Mimicgen, Robosuite, Robomimic must be added to PYTHONPATH in addition to this top level directory

In Ubuntu:

```

export PYTHONPATH="/path/to/thisDirectory:/path/to/robosuite:/path/to/mimicgen:/path/to/robomimic:$PYTHONPATH" 

```
 <a id="dir-struct"></a>
### Directory Structure
**assets:** meshes, texture, and stl files for custom objects  
**custom_devices:** Custom controller device definitions  
**custom_env:** Mimicgen environment definitions  
**custom_subtasks:** Mimicgen config for subtask definitions  
**custom_tasks:** Robosuite task/env definitions  
**scripts:** executables to run environments or record data  
**util:** helper scripts for quicker iteration  

Some data exists within robosuite or mimicgen structure, not currently published here.  
- Custom device control (gello) is added as a selectable option in Robosuite's collect_human_demonstrations.  
- Mimicgen places custom task configs with all other configs in mimicgen/exps.  



### Generating Data

#### 1. Collect Teleoperated demonstrations  
Utilize Robosuite's collect_human_demonstrations script

#### 2. Annotate demos with additional info
Utilize Mimicgen's prepare_src_demonstrations script

#### 3. (if using new task for first time) Update Mimicgen config list
Utilize Mimicgen's generate_config_template script

#### 4. Generate synthetic demos
Utilize Mimicgen's generate_dataset


**If receiving mesa-loader errors in Ubuntu, preload and run script:**

```

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux│ -gnu/libGL.so.1" python...

```


1) 


