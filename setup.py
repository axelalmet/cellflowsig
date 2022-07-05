from setuptools import setup, find_packages

setup(
    name='cellflowsig',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    description='Use graphical causal modeling to infer causal signaling networks from single-cell transcriptomics data and CCC inference.',
    author='Axel A. Almet',
    author_email='axelalmet@gmail.com'
)
