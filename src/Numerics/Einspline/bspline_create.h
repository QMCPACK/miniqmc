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

#ifndef BSPLINE_CREATE_H
#define BSPLINE_CREATE_H

#include "bspline_base.h"
#include "bspline_structs.h"
#include <inttypes.h>


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////              Spline helper functions               ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void find_coefs_1d_d(Ugrid grid, BCtype_d bc, double *data, intptr_t dstride,
                     double *coefs, intptr_t cstride);

void find_coefs_1d_s(Ugrid grid, BCtype_s bc, float *data, intptr_t dstride,
                     float *coefs, intptr_t cstride);


#endif
