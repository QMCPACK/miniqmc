#  Copyright (c) 2019      Peter Doak
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#######################################################################
# These settings control how jobs are launched and results collected
#######################################################################
# the name used to ssh into the machine
set(PYCICLE_MACHINE "local")
# the root location of the build/test tree on the machine
set(PYCICLE_ROOT "/home/epd/CI/miniqmc")
# a flag that says if the machine can send http results to cdash
set(PYCICLE_HTTP TRUE)
set(PYCICLE_COMPILER_TYPE "gcc" )
set(PYCICLE_BUILD_TYPE "Release")

# These versions are ok for gcc or clang
set(BOOST_VER            "1.69.0")
set(CMAKE_VER            "3.14.0")

set(GCC_VER             "7.3.0")
set(PYCICLE_BUILD_STAMP "power9_CUDA10_1-gcc-${GCC_VER}")
set(CFLAGS           "")
set(CXXFLAGS         "")
set(CUDA_DIR         "/usr/local/cuda-10.1")
set(LDFLAGS          "-L/home/epd/spack/opt/spack/linux-centos7-ppc64le/gcc-4.8.5/gcc-7.3.0-bco5a3lq3pzlot65mqywljwofqhsgxim/lib64 -Wl,-rpath,/home/epd/spack/opt/spack/linux-centos7-ppc64le/gcc-4.8.5/gcc-7.3.0-bco5a3lq3pzlot65mqywljwofqhsgxim/lib64 -L${CUDA_DIR}/lib64 -Wl,-rpath,${CUDA_DIR}/lib64")
set(LDCXXFLAGS       "${LDFLAGS}")

set(PYCICLE_COMPILER_SETUP "
    #
    module load Core/gcc/7.3.0
    module load gcc/7.3.0/git
    module load gcc/7.3.0/boost
    module load cuda/10.1
    module load gcc/7.3.0/cmake
    module load gcc/7.3.0/mpich

    #
    # use openmpi compiler wrappers to make MPI use easy
    export CC=mpicc
    export CXX=mpic++
    export CUDA_DIR=\"${CUDA_DIR}\"
    #
    #export CFLAGS=\"${CFLAGS}\"
    #export CXXFLAGS=\"${CXXFLAGS}\"
    export LDFLAGS=\"${LDFLAGS}\"
    export LDCXXFLAGS=\"${LDCXXFLAGS}\"
  ")

set(CTEST_SITE "LECONTE-${PYCICLE_BUILD_STAMP}")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_TEST_TIMEOUT "600")

set(CTEST_PROJECT_NAME "miniqmc")
set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE "cdash-minimal.ornl.gov")
set(CTEST_DROP_LOCATION "/cdash/submit.php?project=miniqmc")
set(CTEST_DROP_SITE_CDASH TRUE)


set(BUILD_PARALLELISM  "16")

#######################################################################
# The string that is used to drive cmake config step
# ensure options (e.g.FLAGS) that have multiple args are escaped
#######################################################################
#  "\"-DCMAKE_C_FLAGS=${CFLAGS}\" "
    

string(CONCAT CTEST_BUILD_OPTIONS ${CTEST_BUILD_OPTIONS}
    "\"-DCMAKE_CXX_COMPILER=mpic++\" "
    "\"-DCMAKE_C_COMPILER=mpicc\" "
    "\"-DCMAKE_C_FLAGS=${CFLAGS}\" "
    "\"-DCMAKE_CXX_FLAGS=${CXXFLAGS}\" "
    "\"-DCMAKE_EXE_LINKER_FLAGS=${LDFLAGS}\" "
    "\"-DCMAKE_BUILD_TYPE=Release\" "
    "\"-DQMC_USE_CUDA=1\" "
    "\"-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_DIR}\" "
    "\"-DCUDA_NVCC_FLAGS=-std=c++14;-arch=sm_60;-Drestrict=__restrict__;-DNO_CUDA_MAIN;-O3;--default-stream per-thread\" "
    "\"-DENABLE_TIMERS=1\" "
    )

#######################################################################
# Setup a slurm job submission template
# note that this is intentionally multiline
#######################################################################
set(PYCICLE_JOB_SCRIPT_TEMPLATE "#!/bin/bash
#PBS -S /bin/bash
#PBS -m be
#PBS -N miniqmc-${PYCICLE_PR}-${PYCICLE_BUILD_STAMP}
#PBS -l nodes=1:ppn=36:gpu_p100
#PBS -l walltime=02:00:00
#PBS -q	gpu_p100
#PBS -A ccsd
#PBS -W group_list=cades-ccsd
#PBS -l qos=std
#PBS -l naccesspolicy=singlejob

# ---------------------
# unload or load modules that differ from the defaults on the system
# ---------------------

#
# ---------------------
# setup stuff that might differ between compilers
# ---------------------
${PYCICLE_COMPILER_SETUP}
"
)
