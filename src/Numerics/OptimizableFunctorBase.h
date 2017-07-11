//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    



/** @file OptimizableFunctorBase.h
 * @brief Define a base class for one-dimensional functions with optimizable variables
 */
#ifndef QMCPLUSPLUS_OPTIMIZABLEFUNCTORBASE_H
#define QMCPLUSPLUS_OPTIMIZABLEFUNCTORBASE_H

#include "config.h"

/** Base class for any functor with optimizable parameters
 *
 * Derived classes from OptimizableFunctorBase are called "functor"s and
 * can be used as a template signature for  Jastrow functions.
 * - OneBodyJastroOrbital<FUNC>
 * - TwoBodyJastroOrbital<FUNC>
 * Functor in qmcpack denotes any function which returns a value at a point, e.g.,
 * GTO, STO, one-dimensional splines etc. OptimizableFunctorBase is introduced for
 * optimizations. The virtual functions are intended for non-critical operations that
 * are executed infrequently during optimizations.
 *
 * This class handles myVars of opt_variables_type (Optimize/VariableSet.h). A derived class
 * can insert any number of variables it handles during optimizations, by calling
 * myVars.insert(name,value);
 * Unlike VarList which uses map, VariableSet is serialized in that the internal order is according
 * to insert calls.
 */
struct OptimizableFunctorBase
{
  ///typedef for real values
  typedef OHMMS_PRECISION real_type;
  ///maximum cutoff
  real_type cutoff_radius;
  ///default constructor
  inline OptimizableFunctorBase() {}
  ///virtual destrutor
  virtual ~OptimizableFunctorBase() {}
};

#endif

