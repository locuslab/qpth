from setuptools import find_packages, setup

setup(
    name='qpth',
    version='0.0.1',
    description="TODO",
    author='Brandon Amos',
    author_email='bamos@cs.cmu.edu',
    platforms=['any'],
    license="Apache 2.0",
    url='https://github.com/locuslab/qp.pytorch',
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
    ]
)
