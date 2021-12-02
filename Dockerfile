ARG IMAGE=nvidia/cuda:11.4.2-devel-ubuntu20.04
FROM ${IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

COPY . /blade
WORKDIR /blade

RUN apt-get install -y g++-10 libspdlog-dev python3-pip cmake ccache
RUN python3 -m pip install meson ninja

ENV CC=gcc-10
ENV CXX=g++-10

RUN rm -fr build
RUN meson build -Dprefix=/usr
RUN cd build && ninja install

WORKDIR /
