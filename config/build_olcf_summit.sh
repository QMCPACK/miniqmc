################################################
# Configuration script for Summit at OLCF      #
# Last modified: Sep 30, 2019                  #
################################################

#IBM XL compiler
module load xl

folder=build_summit_offload_xl
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=mpixlC \
      -D QMC_MPI=1 -D ENABLE_OFFLOAD=1 ..
make -j24
cd ..

folder=build_summit_offload_xl_MP
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=mpixlC -D QMC_MIXED_PRECISION=1 \
      -D QMC_MPI=1 -D ENABLE_OFFLOAD=1 ..
make -j24
cd ..
