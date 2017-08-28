#!/bin/bash -x

BUILD_DIR=$(pwd)
echo $BUILD_DIR

cat > $BUILD_TAG.pbs << EOF
#PBS -A MAT151
#PBS -N $BUILD_TAG
#PBS -j oe
#PBS -l walltime=1:00:00,nodes=1
#PBS -d $BUILD_DIR
#PBS -l partition=rhea

cd $BUILD_DIR

source /sw/rhea/environment-modules/3.2.10/rhel6.7_gnu4.4.7/init/bash

module unload PE-intel 
module load PE-gnu/5.3.0-1.10.2
module load git

env

module list

cd build 

cmake -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" -DCMAKE_CXX_FLAGS="-std=c++11" -DBLAS_blas_LIBRARY="/usr/lib64/libblas.so.3" -DLAPACK_lapack_LIBRARY="/usr/lib64/atlas/liblapack.so.3" ..

make

echo
echo checking J1
echo ----------------------------------------------------
echo

./bin/check_wfc -f J1

echo
echo checking J2
echo ----------------------------------------------------
echo

./bin/check_wfc -f J2

echo
echo checking JeeI
echo ----------------------------------------------------
echo

./bin/check_wfc -f JeeI

echo
echo checking Spline SPO
echo ----------------------------------------------------
echo

./bin/check_spo

EOF

cp $BUILD_TAG.pbs $BUILD_DIR

cd $BUILD_DIR

source scl_source enable rh-python35
which python 

$BUILD_DIR/../../../scripts/blocking_qsub.py $BUILD_DIR $BUILD_TAG.pbs

cp $BUILD_DIR/$BUILD_TAG.o* ../

# get status from checks
[ $(grep 'All checking pass!' ../$BUILD_TAG.o* | wc -l) -eq 4 ]
