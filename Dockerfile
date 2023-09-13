# TODO: Update this Dockerfile.
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing

COPY . /blade
WORKDIR /blade

RUN apt-get install -y g++-10 python3-pip cmake ccache liberfa-dev git libbenchmark-dev libhdf5-dev
RUN python3 -m pip install meson ninja numpy astropy pandas

ENV CC=gcc-10
ENV CXX=g++-10

RUN rm -fr build
RUN git submodule update --init
RUN meson build -Dprefix=/usr
RUN cd build && ninja install

WORKDIR /
