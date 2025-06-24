# Environment Setup:

**mimicgen, robosuite, gello_software must be added to PYTHONPATH in addition to this top level directory**

**In Ubuntu:**

```

export PYTHONPATH="/path/to/thisDirectory:/path/to/robosuite:/path/to/mimicgen:/path/to/gello_software:$PYTHONPATH" 

```

### **--> assets:**

meshes, texture, and stl files for custom objects


### **--> custom_devices:**

Custom controller device definitions
- inherit from Robosuite Device class

### **--> custom_env:**

Robosuite interface subclassed task environments
- inherit from Robosuite RobosuiteInterface class

### **--> custom_subtasks:**

mimicgen config for subtask definitions
- inherit from mimicgen MG_Config class

### **--> custom_tasks:**

Robosuite task definitions
- inherit from Robosuite SingleArmEnv or TwoArmEnv classes

### **--> scripts:**

executables to run environments or record data

### **--> util:**

helper scripts for quicker iteration


If receiving mesa-loader errors in Ubuntu, preload and run script: 

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux│ -gnu/libGL.so.1" python...



