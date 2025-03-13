from setuptools import setup, find_packages

setup(
    name='gym_narde',
    version='0.0.1',
    packages=['gym_narde', 'gym_narde.envs'],
    install_requires=['gymnasium>=1.1.1', 'numpy']
)