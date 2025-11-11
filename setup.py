from setuptools import setup, find_packages

setup(
    name="mimicgen-benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "robosuite=1.4.0",
        "mimicgen>=0.1.0",
        "numpy",
        "trimesh",
        "pymeshlab",
    ],
    package_data={
        "": ["assets/**/*"],
    },
    entry_points={
        "console_scripts": [
            
        ],
    },
)