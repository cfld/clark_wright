#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n routing_env python=3.7 pip
source activate routing_env

pip install scipy
pip install numpy
pip install tqdm

cd /Users/bjohnson/software/tsplib95
pip install -e .