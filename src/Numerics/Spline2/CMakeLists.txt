
  ADD_EXECUTABLE(testSpline2 ../../test/catch-main.cpp test/test_MultiBsplineCUDA.cpp test/test_BsplineAllocatorCUDA.cpp)
  target_link_libraries(testSpline2 qmc_numerics)
