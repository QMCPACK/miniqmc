################################################
# Configuration script for Cray compiler       #
# Last modified: Nov 17, 2019                  #
################################################

module load craype-accel-nvidia60

folder=build_cray_offload
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=CC \
      -D QMC_MPI=1 ..
make -j 16
cd ..

folder=build_cray_offload_MP
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=CC \
      -D QMC_MIXED_PRECISION=1 \
      -D QMC_MPI=1 ..
make -j 16
cd ..
