# coding=utf-8
import glob

from setuptools import setup, find_packages

from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))
base = here


with open(path.join(base, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="Yowie Benchmarks",
    version="0.1",
    install_requires=['docutils>=0.3',
            'setuptools',
            # MRJ: Work out version requirements for these packages
            'numpy',
            'scipy',
            'gym>=0.10.0',
            'h5py',
            'pybullet',
            'tarski',
            'dilithium',
            # MRJ: I think that requiring ipywidgets here is a good idea
            'ipywidgets'
        ],
    packages=find_packages('src'),  # include all packages under src
    package_dir={'': 'src'},  # tell distutils packages are under src

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        'core': ['*.json', '*.png', '*.md', '*.h5'],
    },

    entry_points={
        'console_scripts': [
            # for any executable scripts
        ],
        'gui_scripts': [

        ]
    },

    # metadata to display on PyPI
    author="Miquel Ramirez",
    author_email="miquel.ramirez@unimelb.edu.au",
    description="",
    keywords='ai optimal-control reinforcement-learning planning',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    url="https://github.com/miquelramirez/yowie-benchmarks",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/miquelramirez/yowie-benchmarks",
        "Documentation": "https://github.com/miquelramirez/yowie-benchmarks",
        "Source Code": "https://github.com/miquelramirez/yowie-benchmarks",
    }

    # could also include long_description, download_url, classifiers, etc.
)
