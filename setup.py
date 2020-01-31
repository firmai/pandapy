from setuptools import setup, Command
import os
import sys

setup(name='pandapy',
      version='2.2',
      description='Structured Numpy with Pandas a Click Away',
      url='https://github.com/firmai/pandapy',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['pandapy'],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'numpy-groupies',
          'numba',
          'datetime',
          'ipython'

      ],
      zip_safe=False)
