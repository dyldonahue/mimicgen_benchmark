# MimicGen Benchmark

Custom manipulation tasks, objects, and environments for [RoboSuite](https://github.com/ARISE-Initiative/robosuite) and [MimicGen](https://github.com/NVlabs/mimicgen).

## Overview

This repository extends RoboSuite and MimicGen with custom manipulation tasks featuring:

- **Custom Objects**: Complex, 3rd party CAD with proper physics 
- **Complex Manipulation Tasks**: Expanding MimicGen's base suite of tasks
- **MimicGen Integration**: (some) D0-D2 task variations for data generation

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install RoboSuite, MimicGen and Robomimimic
# As of lat update, no greater than Robosuite=1.4.1 was compatable with MimicGen
https://robosuite.ai/docs/installation.html
https://mimicgen.github.io/docs/introduction/installation.html
https://robomimic.github.io/docs/introduction/installation.html
```

### Install This Package

```bash
# Clone the repository
git clone https://github.com/dyldonahue/mimicgen_benchmark.git
cd mimicgen_benchmark

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "from custom_env.custom import MG_MugTree; print('Installation successful!')"
```

## Quick Start - Data Generation

1. Collect Human demonstrations with task: 

```
python robosuite/robosuite/scripts/collect_human_demonstrations.py --environment env_name --directory path/to/teleop/demos

```

2. Make it Robomimic compatible:

```
python robomimic/robomimic/scripts/conversion/convert_robosuite.py --dataset path/to/teleop/demo/demo.hdf5

```
3. (If you've edited the Mimicgen Task config) Reload Mimicgen Configs:
```
python mimicgen/mimicgen/scripts/generate_config_template.py

```

4. Annote with subtask information:
```
python mimicgen/mimicgen/scripts/prepare_src_dataset.py \
    --dataset path/to/teleop/demo/demo.hdf5 \
    --env_interface_type robosuite \
    --env_interface MG_InterfaceName \
    --output path/to/formatted/demos/demo.hdf5

```
5. Generate Data:
```
python mimicgen/mimicgen/scripts/generate_dataset.py \
    --config mimicgen/mimicgen/exps/templates/robosuite/env_name.json \
    --source path/to/formatted/demos/demo.hdf5 \
    --folder path/to/generated/data \
    --task_name env_name \
    --num_demos 1000

```


## Project Structure

```
mimicgen_benchmark/
├── assets/                     # 3D assets
│   ├── meshes/                # Source STL files
│   ├── objects/               # Processed objects (XML)
│   └── textures/              # Textures for rendering
│
├── custom_tasks/              # RoboSuite task environments
│   ├── mug_tree.py
│   ├── place_on_shelf.py
│   ├── screw_touch.py
│   └── etc
│
├── custom_env/                # MimicGen interface classes
│   └── custom.py              # MG_* wrapper classes
│
├── custom_objects/            # Custom object definitions
│   └── objects.py             # Object classes
│
├── custom_subtasks/           # Subtask definitions for MimicGen 
│   └── configs.py
│
├── scripts/                   # Utility scripts

```
