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

echo ""
echo ""
echo "starting new test for real full precision"
echo ""
echo ""


EOF

/home/bgl/blocking_qsub $BUILD_DIR $BUILD_TAG.pbs

cp $BUILD_DIR/$BUILD_TAG.o* ../

# get status from all checks
[ $(grep -e 'All checks passed for J[123]' -e 'All checks passed for spo' -e 'All checks passed for determinant' ../$BUILD_TAG.o* | wc -l) -eq 9 ]
