FROM ubuntu:24.04

LABEL maintainer="noriki.mo" \
      version="1.0" \
      description="Build and install aarch64-linux-gcc that can be used for contest-2024"

SHELL ["/bin/bash", "-c"]

# Update

RUN apt update && apt upgrade -y

# Pre-process

ENV WORK_PATH=/work

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
    mkdir build && \
    cd build && \
    ../configure \
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
    make INSTALL_HDR_PATH=${GCC_PREFIX} headers_install

## C/C++ Compilers

RUN wget https://ftp.gnu.org/gnu/gcc/gcc-14.1.0/gcc-14.1.0.tar.gz && \
    tar -xf gcc-14.1.0.tar.gz && \
    cd gcc-14.1.0 && \
    mkdir build && \
    cd build && \
    ../configure \
        --prefix=${GCC_PREFIX} \
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
    mkdir build && \
    cd build && \
    ../configure \
        --prefix=${GCC_PREFIX} \
        --with-headers=${GCC_PREFIX}/include \
        --build=${MACHTYPE} \
        --disable-nls \
        --disable-multilib \
        libc_cv_forced_unwind=yes && \
    make install-bootstrap-headers=yes install-headers && \
    make -j$(nproc) csu/subdir_lib && \
    install csu/crt1.o csu/crti.o csu/crtn.o ${GCC_PREFIX}/lib && \
    gcc -nostdlib -nostartfiles -shared -x c /dev/null -o ${GCC_PREFIX}/lib/libc.so && \
    touch ${GCC_PREFIX}/include/gnu/stubs.h

## Compiler Support Library

RUN cd gcc-14.1.0 && \
    cd build && \
    make -j$(nproc) all-target-libgcc && \
    make install-target-libgcc

## Standard C Library

RUN cd glibc-2.38 && \
    cd build && \
    make -j$(nproc) && \
    make install

## Standard C++ Library

RUN cd gcc-14.1.0 && \
    cd build && \
    make -j$(nproc) all && \
    make install

## Confirm

RUN g++ --version

## Update environment variables

ENV GCC_PREFIX=

# CMake >= 3.28

RUN apt update && apt install -y cmake

RUN cmake --version

# Post-process

ENV WORK_PATH=
