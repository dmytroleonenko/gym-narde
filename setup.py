from setuptools import setup, find_packages

setup(
    name="gym_narde",
    version="0.0.1",
    packages=["gym_narde", "gym_narde.envs"],
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.20.0",
        "gymnasium>=1.1.1",
        "pygame",
        "matplotlib",
        "tqdm",
        "psutil",
    ],
    description="An implementation of Narde game for OpenAI Gym with MuZero training",
)