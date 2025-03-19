from setuptools import setup, find_packages

setup(
    name="narde_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "torch",
        "tqdm",
        "matplotlib",
    ],
    author="Narde Team",
    author_email="narde@example.com",
    description="Reinforcement learning tools for the Narde game",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 