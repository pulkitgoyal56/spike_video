from setuptools import setup, find_packages

setup(
    name = 'Spike Video',
    url = 'https://github.com/pulkitgoyal56/spike_video',
    description = ('Synchronises Electrophysiology Spikes to Video.'),
    long_description = open('README.md').read(),
    author = 'Pulkit Goyal @ Burgalossi Lab, Universität Tübingen',
    author_email='pulkitmds@gmail.com',
    version = '0.0.1',
    packages=find_packages(exclude=['tests*']),
)
