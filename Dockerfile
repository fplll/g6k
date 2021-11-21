## -*- docker-image-name: "fplll/g6k" -*-

FROM fplll/fpylll
MAINTAINER Martin Albrecht <fplll-devel@googlegroups.com>

ARG BRANCH=master
ARG JOBS=2
ARG CXXFLAGS="-O2 -march=x86-64"
ARG CFLAGS="-O2 -march=x86-64"

SHELL ["/bin/bash", "-c"]
ENTRYPOINT /usr/local/bin/ipython

RUN apt update && \
    apt install -y python3-scipy && \
    apt clean

RUN git clone --branch $BRANCH https://github.com/fplll/g6k && \
    cd g6k && \
    pip3 install -r requirements.txt && \
    cd ..

RUN cd g6k && \
    CFLAGS=$CFLAGS CXXFLAGS=$CXXFLAGS python3 setup.py build -j $JOBS && \
    python3 setup.py -q install && \
    make clean && \
    cd .. 
