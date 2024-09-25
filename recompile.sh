#!/usr/bin/env bash

make clean
./configure CXX=/usr/bin/g++
python setup.py build_ext --inplace