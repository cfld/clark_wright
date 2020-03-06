#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n cw_env python=3.7 pip
source activate cw_env

pip install scipy
pip install numpy
pip install tqdm
pip install pandas
pip install git+https://github.com/bkj/tsplib95

pip install matplotlib
pip install git+https://github.com/bkj/rsub