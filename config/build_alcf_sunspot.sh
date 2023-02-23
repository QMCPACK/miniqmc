################################################
# Configuration script for icpx compiler       #
# Last modified: Feb 23, 2023                  #
################################################

folder=build_icpx_offload_sycl
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=icpx \
      -D ENABLE_OFFLOAD=ON -D QMC_ENABLE_SYCL=ON ..
make -j 16
cd ..

folder=build_icpx_offload_sycl_MP
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=icpx \
      -D QMC_MIXED_PRECISION=ON \
      -D ENABLE_OFFLOAD=ON -D QMC_ENABLE_SYCL=ON ..
make -j 16
cd ..
