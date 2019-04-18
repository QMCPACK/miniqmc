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

source /sw/rhea/lmod/7.8.2/rhel7.5_4.8.5/lmod/7.8.2/init/bash

module unload intel
module load gcc/6.2.0
module load openblas/0.3.5
module load git/2.18.0
module load cmake/3.13.4

env

module list

echo ""
echo ""
echo "starting new test for real full precision"
echo ""
echo ""

cd build

cmake -DCMAKE_CXX_COMPILER="mpicxx" ..

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

cmake -DQMC_MIXED_PRECISION=1 -DCMAKE_CXX_COMPILER="mpicxx" ..

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

/home/mat151ci_auser/blocking_qsub $BUILD_DIR $BUILD_TAG.pbs

cp $BUILD_DIR/$BUILD_TAG.o* ../

# get status from all checks
[ $(grep -e 'All checks passed for J[123]' -e 'All checks passed for spo' -e 'All checks passed for determinant' ../$BUILD_TAG.o* | wc -l) -eq 10 ]
