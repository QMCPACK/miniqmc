#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
#//
#// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign   
#//		       Anouar Benali, benali@anl.gov, Argonne National Laboratory
#//		       Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign 
#////////////////////////////////////////////////////////////////////////////////////// 

SET(HFILES
 bspline_base.h bspline_create.h
 bspline.h bspline_structs.h
 bspline_eval_d.h
 multi_bspline.h
 multi_bspline_create.h    multi_bspline_structs.h
 multi_bspline_eval_d.h
 multi_bspline_eval_s.h    multi_bspline_eval_z.h
 nubspline_structs.h nubasis.h nubspline.h nugrid.h
 nubspline_base.h nubspline_create.h

 )

#FOREACH(a ${HFILES})
#  #INSTALL(FILES ${EINSPLINE_HOME}/src/${a} DESTINATION ${PROJECT_BINARY_DIR}/include/einspline)
#  configure_file(${EINSPLINE_HOME}/src/${a} ${PROJECT_BINARY_DIR}/include/einspline/${a} COPYONLY)
#ENDFOREACH()

set(SRCS )

SET(SRCS ${SRCS}
  bspline_create.c
  bspline_data.c
  multi_bspline_create.c
  nubspline_create.c
  nubasis.c
)

#do not compiler c functions
IF(HAVE_SSE41)
  SET(SRCS ${SRCS}
  )
else()
  SET(SRCS ${SRCS}
    )
endif()

IF(HAVE_SSE2)
  set(SRCS ${SRCS}
    )
else()
  SET(SRCS ${SRCS}
    )
endif()


if(HAVE_CUDA)
  SET(SRCS  ${SRCS}
    multi_bspline_create_cuda.cu
    bspline_create_cuda.cu
    )
  CUDA_ADD_LIBRARY(einspline ${SRCS})
else()
  ADD_LIBRARY(einspline ${SRCS})
endif()

#target_link_libraries(time_multi einspline)
#add_dependencies(time_multi ${PROJECT_BINARY_HOME}/include/einspline/bspline.h)
#target_link_libraries(test_multi_single einspline)
