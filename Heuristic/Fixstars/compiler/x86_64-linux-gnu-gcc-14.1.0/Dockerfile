FROM ubuntu:24.04

LABEL maintainer="noriki.mo" \
      version="1.0" \
      description="Build and install aarch64-linux-gcc that can be used for contest-2024"

SHELL ["/bin/bash", "-c"]

# Update

RUN apt update && apt upgrade -y

# Pre-process

ENV WORK_PATH=/work
ENV TARGET="x86_64-linux-gnu"
ENV ARCH="x86_64"

WORKDIR ${WORK_PATH}

# GCC, cf. https://preshing.com/20141119/how-to-build-a-gcc-cross-compiler/

ENV GCC_PREFIX="/opt/gcc/gcc-14.1.0"
ENV PATH="${GCC_PREFIX}/bin:${PATH}"

RUN apt update && apt install -y \
    wget tar build-essential \
    libgmp-dev libmpfr-dev libmpc-dev texinfo \
    gawk bison \
    python3 binutils file rsync

## Binutils

RUN wget https://ftp.gnu.org/gnu/binutils/binutils-2.42.tar.gz && \
    tar -xzf binutils-2.42.tar.gz && \
    cd binutils-2.42 && \
    mkdir build-${TARGET} && \
    cd build-${TARGET} && \
    ../configure \
        --target=${TARGET} \
        --prefix=${GCC_PREFIX} \
        --disable-nls \
        --disable-multilib \
        --enable-languages=c,c++ && \
    make -j$(nproc) && \
    make install

## Linux Kernel Headers

RUN wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.5.13.tar.xz && \
    tar -xvf linux-6.5.13.tar.xz && \
    cd linux-6.5.13 && \
    make ARCH=${ARCH} INSTALL_HDR_PATH=${GCC_PREFIX}/${TARGET} headers_install

## C/C++ Compilers

RUN wget https://ftp.gnu.org/gnu/gcc/gcc-14.1.0/gcc-14.1.0.tar.gz && \
    tar -xf gcc-14.1.0.tar.gz && \
    cd gcc-14.1.0 && \
    mkdir build-${TARGET} && \
    cd build-${TARGET} && \
    ../configure \
        --prefix=${GCC_PREFIX} \
        --target=${TARGET} \
        --disable-libsanitizer \
        --disable-nls \
        --disable-multilib \
        --enable-languages=c,c++ && \
    make -j$(nproc) gcc_cv_libc_provides_ssp=yes all-gcc && \
    make install-gcc

## Standard C Library Headers and Startup Files

RUN wget https://ftp.gnu.org/gnu/libc/glibc-2.38.tar.gz && \
    tar -xzf glibc-2.38.tar.gz && \
    cd glibc-2.38 && \
    mkdir build-${TARGET} && \
    cd build-${TARGET} && \
    CC="${TARGET}-gcc" \
    ../configure \
        --prefix=${GCC_PREFIX}/${TARGET} \
        --with-headers=${GCC_PREFIX}/${TARGET}/include \
        --build=${MACHTYPE} \
        --host=${TARGET} \
        --target=${TARGET} \
        --disable-nls \
        --disable-multilib \
        libc_cv_forced_unwind=yes && \
    make install-bootstrap-headers=yes install-headers && \
    make -j$(nproc) csu/subdir_lib && \
    install csu/crt1.o csu/crti.o csu/crtn.o ${GCC_PREFIX}/${TARGET}/lib && \
    ${TARGET}-gcc -nostdlib -nostartfiles -shared -x c /dev/null -o ${GCC_PREFIX}/${TARGET}/lib/libc.so && \
    touch ${GCC_PREFIX}/${TARGET}/include/gnu/stubs.h

## Compiler Support Library

RUN cd gcc-14.1.0 && \
    cd build-${TARGET} && \
    make -j$(nproc) all-target-libgcc && \
    make install-target-libgcc

## Standard C Library

RUN cd glibc-2.38 && \
    cd build-${TARGET} && \
    make -j$(nproc) && \
    make install

## Standard C++ Library

RUN cd gcc-14.1.0 && \
    cd build-${TARGET} && \
    make -j$(nproc) all && \
    make install

## Confirm

RUN ${TARGET}-g++ --version

## Update environment variables

ENV GCC_PREFIX=

# CMake >= 3.28

RUN apt update && apt install -y cmake

RUN cmake --version

# VTune

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt update && apt install -y intel-oneapi-vtune

ENV PATH=/opt/intel/oneapi/vtune/latest/bin64:${PATH}

# Debug symbols

RUN apt update && apt install -y libc-dbg

# Privilege

RUN apt update && apt install -y libcap2-bin

RUN setcap cap_sys_nice+ep /usr/bin/chrt

# Post-process

ENV WORK_PATH=
ENV TARGET=
ENV ARCH=
