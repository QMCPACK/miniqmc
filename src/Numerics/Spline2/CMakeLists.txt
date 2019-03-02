CUDA_ADD_LIBRARY(spline2_test_kernels test/TestKernel.cu)

ADD_EXECUTABLE(testSpline2 ../../test/catch-main.cpp test/test_MultiBsplineCUDA.cpp test/test_BsplineAllocatorCUDA.cpp test/test_WarpAccess.cpp)
target_link_libraries(testSpline2 qmc_numerics spline2_test_kernels)
