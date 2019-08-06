from setuptools import setup, find_packages
import os

FULL_VERSION = '0.0.0'
GPU = bool(os.getenv('GPU'))
if GPU:
    tensorflow_name = 'tensorflow-gpu>=1.10.1,<=1.13.1'
else:
    tensorflow_name = 'tensorflow'

with open('README.md') as f:
    readme = f.read()

setup(
    name='deep_semantic_segmentation',
    version=FULL_VERSION,
    description='Semantic Segmentation Models',
    url='to be appeared',
    long_description=readme,
    author='Asahi Ushio',
    author_email='to be appeared',
    packages=find_packages(exclude=('test', 'dataset', 'random', 'tfrecord', 'checkpoint')),
    include_package_data=True,
    test_suite='test',
    install_requires=[
        'Pillow',
        'numpy',
        tensorflow_name,
        'scipy>=1.2.0',
        'cython'
        # 'toml>=0.10.0',
        # 'pandas',
        # 'nltk',
        # 'sklearn',
        # 'flask',
        # 'werkzeug',
    ]
)

