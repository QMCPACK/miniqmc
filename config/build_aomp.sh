################################################
# Configuration script for AOMP compiler       #
# Last modified: Sep 30, 2019                  #
################################################

#AMD OMP compiler
AOMP=/usr/lib/aomp/bin/clang++

folder=build_aomp
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER="$AOMP" \
      -D CMAKE_CXX_FLAGS="-Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906" \
      -D OFFLOAD_TARGET="amdgcn-amd-amdhsa" \
      -D CMAKE_FIND_ROOT_PATH=/opt/math-libraries/OpenBLAS/current \
      -D QMC_MPI=0 -D ENABLE_OFFLOAD=1 ..
make -j 16
cd ..
