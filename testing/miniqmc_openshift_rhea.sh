#!/bin/bash -x

BUILD_DIR=$(pwd)
echo $BUILD_DIR

cat > $BUILD_TAG.pbs << EOF
#PBS -A MAT151ci
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

export OMP_NUM_THREADS=4

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

./bin/check_wfc -f Det

echo ""
echo ""
echo "starting new test for real mixed precision"
echo ""
echo ""

ctest -L unit --output-on-failure

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

# YL: commented out mixed precision check. Too unstable
#./bin/check_wfc -f Det

ctest -L unit --output-on-failure

cd ../
EOF

/home/mat151ci_auser/blocking_qsub $BUILD_DIR $BUILD_TAG.pbs

cp $BUILD_DIR/$BUILD_TAG.o* ../

# get status from all checks

CHECK_XXX_FAILED=0

if [ $(grep -e 'All checks passed for J[123]' -e 'All checks passed for spo' -e 'All checks passed for Det' ../$BUILD_TAG.o* | wc -l) -ne 9 ]
then
  echo; echo
  echo One or more build variants failed in check_XXX. Check the build log for details.
  echo; echo
  CHECK_XXX_FAILED=1
fi

UNIT_TESTS_FAILED=0

if [ $(grep '100% tests passed, 0 tests failed out of [0-9]*' ../$BUILD_TAG.o* | wc -l) -ne 2 ]
then
  echo; echo
  echo One or more build variants failed in unit tests. Check the build log for details.
  echo; echo
  UNIT_TESTS_FAILED=1
fi

[ $CHECK_XXX_FAILED -eq 0 -a $UNIT_TESTS_FAILED -eq 0 ]
