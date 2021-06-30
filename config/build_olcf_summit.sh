################################################
# Configuration script for Summit at OLCF      #
# Last modified: Jun 30, 2021                  #
################################################

module load cmake
module load git
module load cuda
module load essl
module load netlib-lapack

# IBM XL compiler
module load xl

folder=build_summit_offload_xl
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=mpixlC \
      -D CMAKE_CXX_FLAGS="-qarch=pwr9 -qxflag=disable__cplusplusOverride -isystem /sw/summit/gcc/6.4.0/include/c++/6.4.0/powerpc64le-none-linux-gnu -qgcc_cpp_stdinc=/sw/summit/gcc/6.4.0/include/c++/6.4.0" \
      -D CMAKE_CXX_STANDARD_LIBRARIES=/sw/summit/gcc/6.4.0/lib64/libstdc++.a \
      -D BLAS_essl_LIBRARY=$OLCF_ESSL_ROOT/lib64/libessl.so \
      -D QMC_MPI=1 -D ENABLE_OFFLOAD=1 ..
make -j24
cd ..

folder=build_summit_offload_xl_MP
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=mpixlC -D QMC_MIXED_PRECISION=1 \
      -D CMAKE_CXX_FLAGS="-qarch=pwr9 -qxflag=disable__cplusplusOverride -isystem /sw/summit/gcc/6.4.0/include/c++/6.4.0/powerpc64le-none-linux-gnu -qgcc_cpp_stdinc=/sw/summit/gcc/6.4.0/include/c++/6.4.0" \
      -D CMAKE_CXX_STANDARD_LIBRARIES=/sw/summit/gcc/6.4.0/lib64/libstdc++.a \
      -D BLAS_essl_LIBRARY=$OLCF_ESSL_ROOT/lib64/libessl.so \
      -D QMC_MPI=1 -D ENABLE_OFFLOAD=1 ..
make -j24
cd ..

# LLVM clang compiler
module load gcc/8.1.1 # use GNU MPI wrapper and libraries
# The llvm module is not provided by OLCF. Customize the following line based on your LLVM compilers and set OMPI_CXX=clang++
module load llvm/master-latest

folder=build_summit_offload_clang
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=mpicxx \
      -D BLAS_essl_LIBRARY=$OLCF_ESSL_ROOT/lib64/libessl.so \
      -D QMC_MPI=1 -D ENABLE_OFFLOAD=1 -D USE_OBJECT_TARGET=ON -DOFFLOAD_ARCH=sm_70 ..
make -j24
cd ..

folder=build_summit_offload_clang_MP
mkdir $folder
cd $folder
cmake -D CMAKE_CXX_COMPILER=mpixlC -D QMC_MIXED_PRECISION=1 \
      -D BLAS_essl_LIBRARY=$OLCF_ESSL_ROOT/lib64/libessl.so \
      -D QMC_MPI=1 -D ENABLE_OFFLOAD=1 -D USE_OBJECT_TARGET=ON -DOFFLOAD_ARCH=sm_70 ..
make -j24
cd ..
