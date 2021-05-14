#!/usr/bin/env bash

# Install FPLLL

git clone https://github.com/fplll/fplll
cd fplll || exit
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" $CONFIGURE_FLAGS
make clean
make -j 4
make install
cd ..

# Install FPyLLL
git clone https://github.com/fplll/fpylll
cd fpylll || exit
pip install Cython
pip install -r requirements.txt
pip install -r suggestions.txt
python setup.py clean
python setup.py build
python setup.py install
cd ..

# Otherwise py.test may fail

rm -rf ./fplll
rm -rf ./fpylll
