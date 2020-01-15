from setuptools import setup

setup(name='pandapy',
      version='0.1',
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
          'operator',
          'itertools',
          'numpy-groupies'

      ],
      zip_safe=False)
