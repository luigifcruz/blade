FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing

#
# This is copy-pasta from the README.md file.
# Update this as the README.md file changes.
#

RUN apt install -y git build-essential pkg-config git cmake
RUN apt install -y python3-dev python3-pip
RUN python3 -m pip install meson ninja
RUN apt install -y liberfa-dev libhdf5-dev
RUN apt install -y libbenchmark-dev libgtest-dev
RUN python3 -m pip install numpy astropy pandas

###

COPY . /blade
WORKDIR /blade

RUN rm -fr build
RUN git submodule update --init --recursive
RUN meson build -Dprefix=/usr
RUN cd build && ninja install

WORKDIR /
