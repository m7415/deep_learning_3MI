# setup.py

from setuptools import setup

setup(
    name='my_lib',
    version='0.0.1',
    description='My private library',
    author='Me',

    # Packages
    packages=['my_lib'],
    # Include additional files into the package
    include_package_data=True,
    # Dependent packages (distributions)
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'tensorflow',
        'keras',
        'pydot'
    ],
    python_requires='>=3.6',
)