INCLUDE("${PROJECT_SOURCE_DIR}/CMake/macros.cmake")

IF(QMC_USE_CUDA OR QMC_USE_CLANG_CUDA)
IF(QMC_USE_CUDA)
  CUDA_ADD_LIBRARY(spline2_test_kernels test/TestKernel.cu)
ELSEIF(QMC_USE_CLANG_CUDA)
  ADD_LIBRARY(spline2_test_kernels test/TestKernel.cu)
  target_link_libraries(spline2_test_kernels cudart)
ENDIF(QMC_USE_CUDA)

  ADD_EXECUTABLE(testSpline2 ../../test/catch-main.cpp test/test_MultiBsplineCUDA.cpp test/test_BsplineAllocatorCUDA.cpp test/test_WarpAccess.cpp)
target_link_libraries(testSpline2 qmc_numerics spline2_test_kernels)

IF(QMC_USE_CLANG_CUDA)
  target_link_libraries(testSpline2 cudart)
ENDIF(QMC_USE_CLANG_CUDA)

ADD_UNIT_TEST(testSpline2 "${qmcpack_BINARY_DIR}/bin/testSpline2")

ENDIF(QMC_USE_CUDA OR QMC_USE_CLANG_CUDA)


