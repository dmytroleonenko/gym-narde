from setuptools import setup, find_packages

setup(
    name="muzero-narde",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.20.0",
        "gym-narde",
        "pygame",
        "matplotlib",
        "tqdm",
        "psutil",
    ],
    description="An implementation of MuZero for the Narde game",
    author="Narde Team",
)