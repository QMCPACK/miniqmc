#!/bin/bash
#
# Setup for bora.alcf.anl.gov
#
# Run the nightlies
# 

export TEST_SITE_NAME=bora.alcf.anl.gov
export N_PROCS_BUILD=24
export N_PROCS=32

#Must be an absolute path
place=/sandbox/MINIQMC_CI_BUILDS_DO_NOT_REMOVE

#define and load compiler
compiler=Intel2018

if [ ! -e $place ]; then
mkdir $place
fi

if [ -e $place ]; then
cd $place

echo --- Hostname --- $HOSTNAME
echo --- Checkout for $sys `date`

branch=develop
entry=miniqmc-${branch}

if [ ! -e $entry ]; then
echo --- Cloning miniQMC git `date`
git clone https://github.com/QMCPACK/miniqmc.git $entry
else
echo --- Updating local miniQMC git `date`
cd $entry
git pull
cd ..
fi

if [ -e $entry/CMakeLists.txt ]; then
cd $entry

git checkout $branch

for sys in Real Real-Mixed # Complex-SoA Complex-Mixed-SoA
do

folder=build_$compiler_$sys

if [ -e $folder ]; then
rm -r $folder
fi
mkdir $folder
cd $folder

echo --- Building for $sys `date`

# create log file folder if not exist
mydate=`date +%y_%m_%d`
if [ ! -e $place/log/$entry/$mydate ];
then
  mkdir -p $place/log/$entry/$mydate
fi

CTEST_FLAGS="-DCMAKE_CXX_COMPILER=icpc;-DCMAKE_CXX_FLAGS=-xCOMMON-AVX512"

if [[ $sys == *"Complex"* ]]; then
  CTEST_FLAGS="$CTEST_FLAGS;-DQMC_COMPLEX=1"
fi

if [[ $sys == *"-Mixed"* ]]; then
  CTEST_FLAGS="$CTEST_FLAGS;-DQMC_MIXED_PRECISION=1"
fi

export MINIQMC_TEST_SUBMIT_NAME=${compiler}-${sys}-Release

ctest -DCMAKE_CONFIGURE_OPTIONS=$CTEST_FLAGS -S $PWD/../CMake/ctest_script.cmake -VV --timeout 800 &> $place/log/$entry/$mydate/${MINIQMC_TEST_SUBMIT_NAME}.log

cd ..
echo --- Finished $sys `date`
done

else
echo  "ERROR: No CMakeLists. Bad git clone."
exit 1
fi

else
echo "ERROR: No directory $place"
exit 1
fi
