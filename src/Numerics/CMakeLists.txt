SET(NUMERICS_SRC
    Spline2/einspline_allocator.cpp
    Spline2/bspline_allocator.cpp
    Spline2/MultiBsplineData.cpp
    )

  SET(NUMERICS_CUDA
    Spline2/BsplineAllocatorCUDA.cpp Spline2/MultiBsplineCUDA.cpp
        Spline2/BsplineAllocatorCUDA.cpp Spline2/MultiBsplineFuncsCUDA.cpp
    )

ADD_LIBRARY(qmc_numerics ${NUMERICS_SRC} ${NUMERICS_CUDA})

IF(QMC_USE_CUDA)
  CUDA_ADD_LIBRARY(cuda_numerics Einspline/multi_bspline_cuda_eval_d.cu Einspline/multi_bspline_cuda_eval_s.cu Einspline/multi_bspline_create_cuda.cu)
  target_link_libraries(qmc_numerics cuda_numerics qmc_cuda_params)
ENDIF(QMC_USE_CUDA)


add_subdirectory(Einspline)
add_subdirectory(Spline2)
