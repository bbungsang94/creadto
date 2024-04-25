# Copyright (c) Creadto, Inc. and its affiliates.

import torch
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [2, 1], "Requires PyTorch >= 2.1"

setup(
    name="creadto",
    version="0.5",
    author="Simon Anderson",
    url="www.creadto.com",
    description="Creadto AI Research library for Computer vision and Computer graphics",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
    ],
)