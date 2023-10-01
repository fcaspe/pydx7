from setuptools import setup, find_packages

setup(name='pydx7', version='0.1', packages=find_packages(),
      install_requires=['einops>=0.7.0',
                        'numba>=0.57.0',
                        'numpy>=1.23.5',
                        'setuptools>=65.6.3'])