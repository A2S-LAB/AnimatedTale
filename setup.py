# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name='animated_drawings',
    description="Companion code for `A Method For Automatically Animating Children's Drawings of the Human Form.`",
    author='FAIR',
    author_email='jesse.smith@meta.com',
    python_requires='>=3.8.13',
    install_requires=[
        'numpy== 1.23.3',
        'scipy==1.9.3',
        'scikit-image==0.19.3',
        'scikit-learn==1.1.2',
        'shapely==1.8.5.post1',
        'opencv-python==4.6.0.66',
        'Pillow==9.3.0',
        'glfw==2.5.5',
        'PyOpenGL==3.1.6',
        'PyYAML==6.0',
        'requests==2.31.0',
        'torchserve==0.7.0',
        'tqdm==4.64.1',
        'Flask==2.3.2',
        'python-multipart==0.0.6',
        'diffusers==0.21.4',
        'openai==0.28.1',
        'torchvision==0.16.0',
        'torch==2.1.0',
        'segment-anything==1.0',
        'fastapi==0.104.1',
        'uvicorn==0.23.2'
    ],
    packages=find_packages(),
)
