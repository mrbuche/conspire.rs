#!/bin/sh
# Builds the PTX that the library loads, with the nightly toolchain of rust-toolchain.toml and
# the llvm-bitcode-linker component: rustup component add llvm-bitcode-linker --toolchain nightly
set -e
cd "$(dirname "$0")"
cargo build --release
cp target/nvptx64-nvidia-cuda/release/conspire_gpu_kernels.ptx ../src/domain/gpu/kernels.ptx
