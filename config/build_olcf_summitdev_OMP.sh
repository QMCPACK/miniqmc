################################################
# Configuration script for Summitdev at OLCF   #
# Last modified: Oct 25, 2017                  #
################################################

#IBM compiler
module load xl

folder=build_miniapps_offload_xl
mkdir $folder
cd $folder
cmake -D CMAKE_C_COMPILER="xlc_r" -D CMAKE_CXX_COMPILER="xlc++_r" \
      -D CMAKE_C_COMPILER_ID='XL' -D CMAKE_CXX_COMPILER_ID='XL' \
      -D QMC_MPI=0 -D ENABLE_OFFLOAD=1 ..
make -j24
cd ..

folder=build_miniapps_nooffload_xl
mkdir $folder
cd $folder
cmake -D CMAKE_C_COMPILER="xlc_r" -D CMAKE_CXX_COMPILER="xlc++_r" \
      -D CMAKE_C_COMPILER_ID='XL' -D CMAKE_CXX_COMPILER_ID='XL' \
      -D QMC_MPI=0 ..
make -j24
cd ..

module unload xl

# Clang-ykt
module load clang
# currently not working with CUDA 9
module load cuda/8.0.54

folder=build_miniapps_offload_clang
mkdir $folder
cd $folder
cmake -D CMAKE_C_COMPILER="clang" -D CMAKE_CXX_COMPILER="clang++" \
      -D QMC_MPI=0 -D ENABLE_OFFLOAD=1 ..
make -j24
cd ..

folder=build_miniapps_nooffload_clang
mkdir $folder
cd $folder
cmake -D CMAKE_C_COMPILER="clang" -D CMAKE_CXX_COMPILER="clang++" \
      -D QMC_MPI=0 ..
make -j24
cd ..
