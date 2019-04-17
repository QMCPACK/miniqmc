#!/bin/bash -ex
cmake $HOME/src/miniqmc \
-DQMC_USE_KOKKOS=1 \
-DKOKKOS_PREFIX=$HOME/src/kokkos \
-DKOKKOS_ENABLE_CUDA=true \
-DKOKKOS_ENABLE_OPENMP=false \
-DKOKKOS_ARCH="Power8;Pascal60" \
-DKOKKOS_ENABLE_CUDA_UVM=true \
-DKOKKOS_ENABLE_CUDA_LAMBDA=true \
-DCMAKE_CXX_COMPILER=$HOME/src/kokkos/bin/nvcc_wrapper \
-DCMAKE_CXX_FLAGS="-Drestrict=__restrict__ -D__forceinline=inline" .. \
2>&1 | tee config_log
