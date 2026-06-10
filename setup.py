"""
pip-installable setup for WiringRobot.ConnectionPointDetector (wiringrobot-cpd).

Development install (editable, changes take effect immediately):
    pip install -e .

Normal install:
    pip install .

After installing, the CLI entry points are available:
    run-connector part.off part.labels
    connection-point-detector part.off --checkpoint best.pt
"""
from setuptools import setup

setup(
    name="wiringrobot-cpd",
    version="0.1.0",
    description="WiringRobot.ConnectionPointDetector: robot-ready 3D connection-point detection from connector meshes (DiffusionNet + YOLOv6)",
    python_requires=">=3.9",
    py_modules=[
        # core post-processing pipeline
        "connector_constants",
        "connector3d",
        "feature_geometry",
        "meshio",
        "visualization",
        "config",
        "run_on_file",
        "infer_pipeline",
        # upstream pipeline stages
        "projection_2d3d",
        "remesh",
        "augment",
        "mesh_utils",
        "coco_labels",
        "instances",
        "metrics",
        "dataset_convert",
        # training / HPO
        "hpo_ray",
        "diffusionnet",
        "meshcnn",
        "yolo6",
        # demos / dev
        "demo_all",
    ],
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        # install with: pip install -e ".[dev]"
        "dev": [
            "pytest>=7.0",
            "black",
            "flake8",
        ],
        # exact convex-hull volume centroid, ARAP, fast nearest-neighbour labels
        # install with: pip install -e ".[geom]"
        "geom": [
            "scipy>=1.8",
        ],
        # hyperparameter optimization with Ray Tune (hpo_ray.py)
        # install with: pip install -e ".[hpo]"
        "hpo": [
            "ray[tune]>=2.9",
        ],
        # DiffusionNet training (diffusionnet.py). Also needs the diffusion-net
        # package: pip install git+https://github.com/nmwsharp/diffusion-net
        # install with: pip install -e ".[diffnet]"
        "diffnet": [
            "torch>=1.11",
            "scipy>=1.8",
        ],
        # STEP -> STL tessellation backend for dataset_convert.py
        # install with: pip install -e ".[cad]"
        # (alternatively use FreeCAD/OpenCASCADE via pythonocc-core or cadquery-ocp)
        "cad": [
            "gmsh>=4.11",
        ],
    },
    entry_points={
        "console_scripts": [
            # allows calling 'run-connector mesh.off mesh.labels' from the terminal
            "run-connector=run_on_file:main",
            # end-to-end inference: 'connector-infer part.off --checkpoint best.pt'
            "connector-infer=infer_pipeline:main",
            # robotic-named alias for the same entry point
            "connection-point-detector=infer_pipeline:main",
        ],
    },
)