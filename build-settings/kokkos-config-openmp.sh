#!/bin/bash -ex
cmake $HOME/src/miniqmc_acangi_2/ \
-DQMC_USE_KOKKOS=1 \
-DKOKKOS_PREFIX=$HOME/src/kokkos \
-DKOKKOS_ARCH="Power8" \
-DKOKKOS_ENABLE_OPENMP=true \
-DCMAKE_CXX_FLAGS="-Drestrict=__restrict__ -D__forceinline=inline" .. \
2>&1 | tee config_log
