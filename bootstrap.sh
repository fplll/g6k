#!/usr/bin/env bash

# Create Virtual Environment

rm -rf g6k-env
virtualenv g6k-env
cat <<EOF >>g6k-env/bin/activate
### LD_LIBRARY_HACK
_OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
### END_LD_LIBRARY_HACK

### PKG_CONFIG_HACK
_OLD_PKG_CONFIG_PATH="\$PKG_CONFIG_PATH"
PKG_CONFIG_PATH="\$VIRTUAL_ENV/lib/pkgconfig:\$PKG_CONFIG_PATH"
export PKG_CONFIG_PATH
### END_PKG_CONFIG_HACK
      
CFLAGS="\$CFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
CXXFLAGS="\$CXXFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
export CFLAGS
export CXXFLAGS
EOF

source g6k-env/bin/activate

pip install -U pip

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
python setup.py build_ext
python setup.py install
cd ..

pip install -r requirements.txt
python setup.py clean
python setup.py build_ext --inplace

# Otherwise py.test may fail

rm -rf ./fplll
rm -rf ./fpylll
