#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
pushd "${SCRIPT_DIR}" >/dev/null

rm -rf build
mkdir -p build
cd build

if [ "$1" == "rocketlake" ]; then
    FIXCON_TARGET_ARCH="rocketlake"
    FIXCON_CXX_COMPILER=x86_64-linux-gnu-g++
elif [ "$1" == "icelake" ]; then
    FIXCON_TARGET_ARCH="icelake-client"
    FIXCON_CXX_COMPILER=x86_64-linux-gnu-g++
elif [ "$1" == "rocketlake-profile" ]; then
    FIXCON_TARGET_ARCH="rocketlake"
    FIXCON_CXX_COMPILER=x86_64-linux-gnu-g++
    FIXCON_CXX_FLAGS="-g"
else
    FIXCON_TARGET_ARCH="native"
    FIXCON_CXX_COMPILER=g++
fi

cmake .. -DTARGET_ARCH=${FIXCON_TARGET_ARCH} -DCMAKE_CXX_COMPILER=${FIXCON_CXX_COMPILER} -DCMAKE_CXX_FLAGS=${FIXCON_CXX_FLAGS}
make

popd >/dev/null

set +e
