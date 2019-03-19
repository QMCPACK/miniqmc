SET(NUMERICS_SRC
    Spline2/einspline_allocator.cpp
    Spline2/bspline_allocator.cpp
    Spline2/BsplineSet.cpp
    Spline2/MultiBsplineData.cpp
    )

  #not sure this is necessary
  SET(NUMERICS_CUDA_OUTSIDE "")
  SET(NUMERICS_CUDA_INSIDE  "")
IF(QMC_USE_CUDA OR QMC_USE_CLANG_CUDA)
  MESSAGE("ADDING CUDA NUMERICS SOURCE FILES")
  SET(NUMERICS_CUDA_OUTSIDE
        Spline2/BsplineAllocatorCUDA.cpp Spline2/BsplineSetCUDA.cpp Spline2/MultiBsplineFuncsCUDA.cpp
	)
  SET(NUMERICS_CUDA_INSIDE Einspline/multi_bspline_cuda_eval_d.cu Einspline/multi_bspline_cuda_eval_s.cu Einspline/MultiBsplineCreateCUDA.cu)
ENDIF(QMC_USE_CUDA OR QMC_USE_CLANG_CUDA)

ADD_LIBRARY(qmc_numerics ${NUMERICS_SRC} ${NUMERICS_CUDA_OUTSIDE})

IF(QMC_USE_CUDA)
  CUDA_ADD_LIBRARY(cuda_numerics ${NUMERICS_CUDA_INSIDE})
ELSEIF(QMC_USE_CLANG_CUDA)
  ADD_LIBRARY(cuda_numerics ${NUMERICS_CUDA_INSIDE})
ENDIF(QMC_USE_CUDA)

if(QMC_USE_CUDA OR QMC_USE_CLANG_CUDA)
target_link_libraries(qmc_numerics cuda_numerics qmc_cuda_params)
endif(QMC_USE_CUDA OR QMC_USE_CLANG_CUDA)

add_subdirectory(Spline2)
