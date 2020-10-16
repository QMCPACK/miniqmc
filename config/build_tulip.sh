module load PrgEnv-cray/1.0.6
module swap gcc gcc/8.1.0
module load hdf5/1.10.1
module load openblas

echo
echo ###################################
echo Building V100_Cray_offload_real_MP
echo ###################################
module load craype-accel-nvidia70
folder=build_V100_Cray_offload_real_MP
mkdir $folder
cd $folder
cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DENABLE_OFFLOAD=1 -DQMC_MIXED_PRECISION=1 -DENABLE_TIMERS=1 -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment -DQMC_ENABLE_CUDA=ON ..
make -j16
cd ..
module unload craype-accel-nvidia70

echo
echo ###################################
echo Building MI60_Cray_offload_real_MP
echo ###################################
module load rocm
module load craype-accel-amd-gfx906
folder=build_MI60_Cray_offload_real_MP
mkdir $folder
cd $folder
cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DENABLE_OFFLOAD=1 -DQMC_MIXED_PRECISION=1 -DENABLE_TIMERS=1 -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment -DQMC_ENABLE_ROCM=ON -DROCM_ROOT=$ROCM_PATH ..
make -j16
cd ..
module unload craype-accel-amd-gfx906
