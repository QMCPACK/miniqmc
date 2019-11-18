################################################
# Configuration script for GCC compiler       #
# Last modified: Nov 17, 2019                  #
################################################

module load gcc mkl

folder=build_gnu_offload
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=g++ \
      -D ENABLE_MKL=1 \
      -D ENABLE_OFFLOAD=1 ..
make -j 16
cd ..

folder=build_gnu_offload_MP
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=g++ \
      -D QMC_MIXED_PRECISION=1 \
      -D ENABLE_MKL=1 \
      -D ENABLE_OFFLOAD=1 ..
make -j 16
cd ..
