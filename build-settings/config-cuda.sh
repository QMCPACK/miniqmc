KOKKOS_ROOT=/home/lshulen/sandbox/kokkos
cmake -DQMC_USE_KOKKOS=1 \
    -DKOKKOS_PREFIX=${KOKKOS_ROOT} \
    -DKOKKOS_ENABLE_CUDA=false \
    -DKOKKOS_ENABLE_OPENMP=true \
    -DKOKKOS_ARCH="HSW" \
    -DKOKKOS_ENABLE_EXPLICIT_INSTANTIATION=false \
    -DCMAKE_CXX_COMPILER="clang++" \
    -DCMAKE_CXX_FLAGS="-Drestrict=__restrict__ -D__forceinline=inline" \
    ..
