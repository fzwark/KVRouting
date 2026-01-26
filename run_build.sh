#!/bin/bash

unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

export NO_PROXY="127.0.0.1,localhost"

cd sglang-0.4.6
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools wheel build setuptools-rust
pip install -e "python[all]"    
pip install protobuf
pip install cython setuptools
python setup.py build_ext --inplace
cd ..

export HOME=$PWD
mkdir -p $HOME/.cache

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd sglang-0.4.6/sgl-router
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env" || true
export PATH="$HOME/.cargo/bin:$PATH"
pip install setuptools-rust wheel build
python -m build
pip install dist/*.whl
cd ../..