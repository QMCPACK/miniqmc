#!/bin/bash -ex
cmake $HOME/src/miniqmc \
-DCMAKE_CXX_COMPILER:FILEPATH=$HOME/src/kokkos/bin/nvcc_wrapper \
-DCMAKE_CXX_FLAGS:STRING="-O3 -g" \
-DCMAKE_BUILD_TYPE:STRING=None \
-DQMC_MPI:BOOL=0 \
-DQMC_USE_KOKKOS:BOOL=ON \
-DKOKKOS_PREFIX:PATH=$HOME/install/kokkos \
2>&1 | tee config_log
