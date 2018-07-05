#!/bin/bash -ex
cmake $HOME/src/kokkos \
-DKOKKOS_ARCH="Power8;Pascal61" \
-DCMAKE_INSTALL_PREFIX=$HOME/install/kokkos \
-DKOKKOS_ENABLE_CUDA:BOOL=ON \
-DKOKKOS_ENABLE_SERIAL:BOOL=ON \
-DCMAKE_CXX_COMPILER=$HOME/src/kokkos/bin/nvcc_wrapper \
-DBUILD_SHARED_LIBS:BOOL=ON \
2>&1 | tee config_log
