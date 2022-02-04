from setuptools import setup


setup(
    name='DNI-RNN',
    version='0.1.0',
    author='Michael Hu',
    author_email='prmhu@yahoo.com',
    url='https://github.com/mtanghu/DNI-RNN',
    description=(
        'Decoupled Neural Interfaces using Synthetic Gradients for PyTorch'
    ),
    py_modules=['dni'],
    install_requires=[
        'torch>=1.0.0'
    ]
)