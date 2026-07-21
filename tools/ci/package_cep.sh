#!/usr/bin/env sh
set -eu

BUILD_TYPE="${BUILD_TYPE:-release}"
BLADE_DEVICES="${BLADE_DEVICES:-auto}"
CEP_OUTPUT_DIR="${CEP_OUTPUT_DIR:-.dist/cep}"

rm -rf build
meson setup build -Dbuildtype="${BUILD_TYPE}" -Ddevices="${BLADE_DEVICES}"
meson compile -C build blade_cep

mkdir -p "${CEP_OUTPUT_DIR}"
cp build/blade.cep "${CEP_OUTPUT_DIR}/blade.cep"
