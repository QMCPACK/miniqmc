/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////

#ifndef MULTI_BSPLINE_CREATE_H
#define MULTI_BSPLINE_CREATE_H

#include "bspline_base.h"
#include "multi_bspline_structs.h"


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////              Spline creation functions             ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

/////////////////////////////////////
// Uniform, single precision, real //
/////////////////////////////////////

// Set the data for the splines, and compute spline coefficients
void set_multi_UBspline_3d_s(multi_UBspline_3d_s *spline, int spline_num,
                             float *data);

void set_multi_UBspline_3d_s_d(multi_UBspline_3d_s *spline, int spline_num,
                               double *data);

/////////////////////////////////////
// Uniform, double precision, real //
/////////////////////////////////////

// Set the data for the splines, and compute spline coefficients
void set_multi_UBspline_3d_d(multi_UBspline_3d_d *spline, int spline_num,
                             double *data);

#endif
