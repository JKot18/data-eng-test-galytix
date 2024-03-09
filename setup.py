from setuptools import setup, find_packages

setup(
    name='data-eng-test',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy ',
        'gensim ',
        'scipy ',
        'tqdm  ',
        'typing',
        'argparse'
    ],
    entry_points={
        'console_scripts': [
            'main = main:main',
        ],
    },
    author='Julia Kot',
    author_email='julia.kot@gmail.com',
    description='Data Engineering task for Galytix'
)
