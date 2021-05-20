################################################
# Configuration script for AOMP compiler       #
# Last modified: Sep 30, 2019                  #
################################################

#AMD OMP compiler
AOMP=/usr/lib/aomp/bin/clang++

folder=build_ryzen_aomp
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER="$AOMP" \
      -D OFFLOAD_TARGET=amdgcn-amd-amdhsa -D OFFLOAD_ARCH=gfx906 \
      -D ENABLE_OFFLOAD=1 ..
make -j 16
cd ..

folder=build_ryzen_aomp_MP
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER="$AOMP" \
      -D OFFLOAD_TARGET=amdgcn-amd-amdhsa -D OFFLOAD_ARCH=gfx906 \
      -D QMC_MIXED_PRECISION=1 -D ENABLE_OFFLOAD=1 ..
make -j 16
cd ..
