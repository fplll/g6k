#!/usr/bin/env bash

jobs="-j 4 "
if [ "$1" = "-j" ]; then
   jobs="-j $2 "
fi

if [ "$PYTHON" = "" ]; then PYTHON=python; export PYTHON; fi
PIP="$PYTHON -m pip"

PYVER=$($PYTHON --version | cut -d' ' -f2)
echo "Usage:"
echo "   ./bootstrap.sh [ -j <#jobs> ]  (uses system's python)"
echo "   PYTHON=python2 ./bootstrap.sh  (uses python2)"
echo "   PYTHON=python3 ./bootstrap.sh  (uses python3)"
echo " "
echo "Using python version: $PYVER"
echo "Using $jobs"
sleep 1

# Create Virtual Environment

rm -rf g6k-env activate
$PYTHON -m virtualenv g6k-env -p $PYTHON
if [ ! -d g6k-env ]; then
    echo "Failed to create virtual environment in 'g6k-env'!"
    echo "Is '$PYTHON -m virtualenv' working?"
    echo "Try '$PIP install virtualenv' otherwise."
    exit 1 # 1 is the exit value if creating virtualenv fails
fi

cat <<EOF >>g6k-env/bin/activate

### LD_LIBRARY_HACK
_OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
### END_LD_LIBRARY_HACK

CFLAGS="\$CFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
CXXFLAGS="\$CXXFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
export CFLAGS
export CXXFLAGS

### PKG_CONFIG_HACK
_OLD_PKG_CONFIG_PATH="\$PKG_CONFIG_PATH"
PKG_CONFIG_PATH="\$VIRTUAL_ENV/lib/pkgconfig:\$PKG_CONFIG_PATH"
export PKG_CONFIG_PATH
### END_PKG_CONFIG_HACK

PYTHON="$PYTHON"
export PYTHON
unalias python 2>/dev/null
unalias pip 2>/dev/null
alias python=$PYTHON
alias pip="$PYTHON -m pip"
EOF

ln -s g6k-env/bin/activate ./
source ./activate
$PIP install -U pip -r requirements.txt

# Install FPLLL

git clone https://github.com/fplll/fplll g6k-fplll
cd g6k-fplll || exit
git pull # Update if it was checked-out before
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" $CONFIGURE_FLAGS

if ! make clean; then
    echo "Make clean failed in fplll. This is usually because there was an error with either autogen.sh or configure."
    echo "Check the logs above - they'll contain more information."
    exit 2 # 2 is the exit value if building fplll fails via configure or autogen
fi

if ! make $jobs; then
    echo "Making fplll failed."
    echo "Check the logs above - they'll contain more information."
    exit 3 # 3 is the exit value if building fplll fails as a result of make $jobs.
fi

if ! make install; then
    echo "Make install failed for fplll."
    echo "Check the logs above - they'll contain more information."
    exit 4 # 4 is the exit value if installing fplll failed.
fi

cd ..

# Install FPyLLL

git clone https://github.com/fplll/fpylll g6k-fpylll
cd g6k-fpylll || exit
git pull # Update if it was checked-out before
$PIP install -r requirements.txt -r suggestions.txt

$PYTHON setup.py clean
if ! $PYTHON setup.py build_ext $jobs || $PYTHON setup.py build_ext; then
    echo "Failed to build FPyLLL!"
    echo "Check the logs above - they'll contain more information."
    exit 5
fi
$PYTHON setup.py install

cd ..

# Build G6K

$PYTHON setup.py clean
if ! $PYTHON setup.py build_ext $jobs --inplace || $PYTHON setup.py build_ext --inplace; then
    echo "Failed to build G6K!"
    echo "Check the logs above - they'll contain more information."
    exit 6
fi

# Fin

echo " "
echo "Don't forget to activate environment each time:"
echo " source ./activate"
echo "This will also add the following aliases:"
grep "^alias" activate
