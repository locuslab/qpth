from setuptools import find_packages, setup

setup(
    name='qpth',
    version='0.0.6',
    description="A fast and differentiable QP solver for PyTorch.",
    author='Brandon Amos',
    author_email='bamos@cs.cmu.edu',
    platforms=['any'],
    license="Apache 2.0",
    url='https://github.com/locuslab/qpth',
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
    ]
)
