## -*- docker-image-name: "fplll/g6k" -*-

FROM fplll/fpylll
MAINTAINER Martin Albrecht <fplll-devel@googlegroups.com>

ARG BRANCH=master
ARG JOBS=2

SHELL ["/bin/bash", "-c"]
ENTRYPOINT /usr/local/bin/ipython

RUN git clone --branch $BRANCH https://github.com/fplll/g6k && \
    cd g6k && \
    make && \
    pip3 install -r requirements.txt && \
    python3 setup.py build && \
    python3 setup.py -q install && \
    make clean
