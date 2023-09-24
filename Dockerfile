# TODO: Update this Dockerfile.
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing

COPY . /blade
WORKDIR /blade

# System dependencies.
RUN apt install -y build-essential pkg-config git python3-pip

# Build dependencies.
RUN python3 -m pip install cmake meson ninja

# Test dependencies.
RUN apt install -y libbenchmark-dev
RUN python3 -m pip install numpy astropy pandas

# ATA phasor module dependencies.
RUN apt install -y liberfa-dev

# HDF5 writer dependencies.
RUN apt install -y libhdf5-dev

RUN rm -fr build
RUN git submodule update --init
RUN meson build -Dprefix=/usr
RUN cd build && ninja install

WORKDIR /
