KOKKOS_ROOT=/eppic/GroupSoftware/kokkos
cmake -DQMC_USE_KOKKOS=1 \
    -DQMC_MIXED_PRECISION=1 \
    -DKOKKOS_PREFIX=${KOKKOS_ROOT} \
    -DKOKKOS_ENABLE_CUDA=true \
    -DKOKKOS_ENABLE_OPENMP=false \
    -DKOKKOS_ARCH="BDW,Pascal60" \
    -DKOKKOS_ENABLE_CUDA_UVM=true \
    -DKOKKOS_ENABLE_CUDA_LAMBDA=true \
    -DKOKKOS_ENABLE_EXPLICIT_INSTANTIATION=false \
    -DCMAKE_CXX_COMPILER=${KOKKOS_ROOT}/bin/nvcc_wrapper \
    -DCMAKE_CXX_FLAGS="-g -Drestrict=__restrict__ -D__forceinline=inline " \
    ..
