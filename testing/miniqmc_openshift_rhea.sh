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
module load cmake/3.6.1

env

module list

echo ""
echo ""
echo "starting new test for real full precision"
echo ""
echo ""

cd build

cmake -DCMAKE_CXX_COMPILER="mpicxx" -DBLAS_blas_LIBRARY="/usr/lib64/libblas.so.3" -DLAPACK_lapack_LIBRARY="/usr/lib64/atlas/liblapack.so.3" ..

make

echo
echo checking J1 full precision
echo ----------------------------------------------------
echo

./bin/check_wfc -f J1

echo
echo checking J2 full precision
echo ----------------------------------------------------
echo

./bin/check_wfc -f J2

echo
echo checking J3 full precision
echo ----------------------------------------------------
echo

./bin/check_wfc -f J3

echo
echo checking Spline SPO full precision
echo ----------------------------------------------------
echo

./bin/check_spo

echo
echo checking Determinant update full precision
echo ----------------------------------------------------
echo

./bin/check_determinant

echo ""
echo ""
echo "starting new test for real mixed precision"
echo ""
echo ""

cd ../
rm -rf ./build
mkdir -p build
cd build

cmake -DQMC_MIXED_PRECISION=1 -DCMAKE_CXX_COMPILER="mpicxx" -DBLAS_blas_LIBRARY="/usr/lib64/libblas.so.3" -DLAPACK_lapack_LIBRARY="/usr/lib64/atlas/liblapack.so.3" ..

make

echo
echo checking J1 mixed precision
echo ----------------------------------------------------
echo

./bin/check_wfc -f J1

echo
echo checking J2 mixed precision
echo ----------------------------------------------------
echo

./bin/check_wfc -f J2

echo
echo checking J3 mixed precision
echo ----------------------------------------------------
echo

./bin/check_wfc -f J3

echo
echo checking Spline SPO mixed precision
echo ----------------------------------------------------
echo

./bin/check_spo

echo
echo checking Determinant update mixed precision
echo ----------------------------------------------------
echo

./bin/check_determinant

EOF

/home/bgl/blocking_qsub $BUILD_DIR $BUILD_TAG.pbs

cp $BUILD_DIR/$BUILD_TAG.o* ../

# get status from all checks
[ $(grep -e 'All checks passed for J[123]' -e 'All checks passed for spo' -e 'All checks passed for determinant' ../$BUILD_TAG.o* | wc -l) -eq 10 ]
