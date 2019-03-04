//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign   
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign 
//////////////////////////////////////////////////////////////////////////////////////


#ifndef MULTI_BSPLINE_CREATE_CUDA_H
#define MULTI_BSPLINE_CREATE_CUDA_H

#include "multi_bspline_structs_cuda.h"

extern "C" multi_UBspline_3d_s<Devices::CUDA>*
create_multi_UBspline_3d_s_cuda (multi_UBspline_3d_s<Devices::CPU>* spline);

extern "C" multi_UBspline_3d_s<Devices::CUDA>*
create_multi_UBspline_3d_s_cuda_conv (multi_UBspline_3d_d<Devices::CPU>* spline);


/* extern "C" multi_UBspline_3d_c<Devices::CUDA>* */
/* create_multi_UBspline_3d_c_cuda (multi_UBspline_3d_c<Devices::CPU>* spline); */

/* extern "C" multi_UBspline_3d_c<Devices::CUDA>* */
/* create_multi_UBspline_3d_c_cuda_conv (multi_UBspline_3d_z<Devices::CPU>* spline); */


multi_UBspline_3d_d<Devices::CUDA>*
create_multi_UBspline_3d_d_cuda (multi_UBspline_3d_d<Devices::CPU>* spline);

/* extern "C" multi_UBspline_3d_z<Devices::CUDA>* */
/* create_multi_UBspline_3d_z_cuda (multi_UBspline_3d_z<Devices::CPU>* spline); */

#endif
