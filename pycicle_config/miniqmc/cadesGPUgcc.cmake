#  Copyright (c) 2019      Peter Doak
#  Copyright (c) 2017-2019 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#######################################################################
# These settings control how jobs are launched and results collected
#######################################################################
# the name used to ssh into the machine
set(PYCICLE_MACHINE "local")
# the root location of the build/test tree on the machine
set(PYCICLE_ROOT "/lustre/or-hydra/cades-cnms/epd/MINIQMC_CI")
# a flag that says if the machine can send http results to cdash
set(PYCICLE_HTTP TRUE)
# Launch jobs using pbs rather than directly running them on the machine
set(PYCICLE_JOB_LAUNCH "pbs")
set(PYCICLE_COMPILER_TYPE "gcc" )
set(PYCICLE_BUILD_TYPE "Release")

# These versions are ok for gcc or clang
set(BOOST_VER            "1.65.0")
set(HWLOC_VER            "1.11.7")
set(JEMALLOC_VER         "5.0.1")
set(OTF2_VER             "2.0")
set(PAPI_VER             "5.5.1")
set(BOOST_SUFFIX         "1_65_0")
set(CMAKE_VER            "3.9.1")

if (PYCICLE_COMPILER_TYPE MATCHES "gcc")
  set(GCC_VER             "6.5.0")
  set(PYCICLE_BUILD_STAMP "CUDA9_2-gcc-${GCC_VER}")
  #
  #set(INSTALL_ROOT     "/apps/daint/UES/6.0.UP04/HPX")
  #
  set(CFLAGS           "-fPIC -march=native -mtune=native -ffast-math")
  set(CXXFLAGS         "-fPIC -march=native -mtune=native -ffast-math")
  set(LDFLAGS          "-L/software/dev_tools/swtree/cs400_centos7.2_pe2016-08/gcc/5.3.0/centos7.2_gcc4.8.5/lib64 -Wl,-rpath,/software/dev_tools/swtree/cs400_centos7.2_pe2016-08/gcc/5.3.0/centos7.2_gcc4.8.5/lib64")
  set(LDCXXFLAGS       "${LDFLAGS}")
  set(FFTW_DIR         "/software/user_tools/centos-7.2.1511/cades-cnms/spack/opt/spack/linux-centos7-x86_64/gcc-6.5.0/fftw-3.3.8-kpdartcqxfk2kdsbcfdtwin75s24z5uu")
  set(HDF5_DIR         "/software/user_tools/centos-7.2.1511/cades-cnms/spack/opt/spack/linux-centos7-x86_64/gcc-6.5.0/hdf5-1.10.4-4gmsnjn7fozngnc3gwckwnoi2dq53yon")
  #set by module load cuda/9.2
  #set(CUDA_DIR         "/software/user_tools/centos-7.2.1511/cades-cnms/spack/opt/spack/linux-centos7-x86_64/gcc-5.3.0/cuda-8.0.61-pz7ileloxiwrc7kvi4htvwo5p7t3ugvv")
  set(MAGMA_DIR        "/software/user_tools/centos-7.2.1511/cades-cnms/spack/opt/spack/linux-centos7-x86_64/gcc-6.5.0/magma-2.4.0-ndhxaftye4ji5bckhwjv23f5rhvrebai")
  # multiline string
  set(PYCICLE_COMPILER_SETUP "
    #
    spack load gcc/egooyqw
    spack load git@2.12.1
    spack load fftw/kpdartc
    spack load cmake/q76ndqk
    spack load mpich/6zgajlw
    spack load hdf5/4gmsnjn
    module load cuda/9.2
    module load magma/ndhxaft
    #
    # use openmpi compiler wrappers to make MPI use easy
    export CC=mpicc
    export CXX=mpic++
    #
    #export CFLAGS=\"${CFLAGS}\"
    #export CXXFLAGS=\"${CXXFLAGS}\"
    export LDFLAGS=\"${LDFLAGS}\"
    export LDCXXFLAGS=\"${LDCXXFLAGS}\"
  ")

elseif(PYCICLE_COMPILER_TYPE MATCHES "clang")
endif()

# set(HWLOC_ROOT       "${INSTALL_ROOT}/hwloc/${HWLOC_VER}")
# set(JEMALLOC_ROOT    "${INSTALL_ROOT}/jemalloc/${JEMALLOC_VER}")
# set(OTF2_ROOT        "${INSTALL_ROOT}/otf2/${OTF2_VER}")
# set(PAPI_ROOT        "${INSTALL_ROOT}/papi/${PAPI_VER}")
# set(PAPI_INCLUDE_DIR "${INSTALL_ROOT}/papi/${PAPI_VER}/include")
# set(PAPI_LIBRARY     "${INSTALL_ROOT}/papi/${PAPI_VER}/lib/libpfm.so")

set(CTEST_SITE "CADES_CONDO-${PYCICLE_BUILD_STAMP}")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_TEST_TIMEOUT "600")
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
    "\"-DCMAKE_EXE_LINKER_FLAGS=-L/software/user_tools/centos-7.2.1511/cades-cnms/spack/opt/spack/linux-centos7-x86_64/gcc-8.2.0/gcc-6.5.0-egooyqwfmyg6msi5xykwsvniotp774yx/lib64 -Wl,-rpath,/software/user_tools/centos-7.2.1511/cades-cnms/spack/opt/spack/linux-centos7-x86_64/gcc-8.2.0/gcc-6.5.0-egooyqwfmyg6msi5xykwsvniotp774yx/lib64\" "
    "\"-DCMAKE_BUILD_TYPE=Release\" "
    "\"-DQMC_USE_CUDA=1\" "
    "\"-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_DIR}\" "
    "\"-DCUDA_NVCC_FLAGS=-std=c++14;-arch=sm_60;-Drestrict=__restrict__;-DNO_CUDA_MAIN;-O3;--default-stream per-thread\" "
    "\"-DENABLE_TIMERS=1\" "
    "\"-DFFTW_INCLUDE_DIR=${FFTW_DIR}/include\" "
    "\"-DFFTW_LIBRARY=${FFTW_DIR}/lib/libfftw3.a\" "
    )

#######################################################################
# Setup a slurm job submission template
# note that this is intentionally multiline
#######################################################################
set(PYCICLE_JOB_SCRIPT_TEMPLATE "#!/bin/bash
#PBS -S /bin/bash
#PBS -m be
#PBS -N DCA-${PYCICLE_PR}-${PYCICLE_BUILD_STAMP}
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
. /software/user_tools/current/cades-cnms/spack/share/spack/setup-env.sh
#
# ---------------------
# setup stuff that might differ between compilers
# ---------------------
${PYCICLE_COMPILER_SETUP}
"
)
