import os

from setuptools import Command, find_packages, setup


class CleanCommand(Command):
    user_options = []  # type: ignore

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


cmdclass = {'clean': CleanCommand}

setup(
    name='stereo',
    packages=find_packages('src'),
    provides=['stereo'],
    license='MIT',
    package_dir={'': 'src'},
    version="0.0.1",
    cmdclass=cmdclass,
    author="Euxhen Hasanaj",
    author_email="ehasanaj@cs.cmu.edu",
    description=("Sparse Temporal Gene Regulatory Networks using Evolving Graph Convolutional Networks"),
    long_description=("Same as above."),
    python_requires=">=3.10",
)
