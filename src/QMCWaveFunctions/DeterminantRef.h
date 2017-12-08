////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2017 QMCPACK developers.
//
// File developed by: M. Graham Lopez
//
// File created by: M. Graham Lopez
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file DeterminantRef.h
 * @brief Determinant piece of the wave function
 */

#ifndef QMCPLUSPLUS_DETERMINANTREF_H
#define QMCPLUSPLUS_DETERMINANTREF_H

namespace qmcplusplus
{

struct DiracDeterminant
{
  // constructor
  explicit DiracDeterminant(int nels)
  {
    /*
    psiMinv.resize(nels,nels);
    psiV.resize(nels);
    psiM.resize(nels,nels);

    pivot.resize(nels);
    psiMsave.resize(nels,nels);

    // now we "void initialize(RandomGenerator<T> RNG)"

    int nels=psiM.rows();
    //get lwork and resize workspace
    LWork=getGetriWorkspace(psiM.data(),nels,nels,pivot.data());
    work.resize(LWork);

    myRandom=RNG;
    constexpr T shift(0.5);
    RNG.generate_uniform(psiMsave.data(),nels*nels);
    psiMsave -= shift;

    INVT phase;
    transpose(psiMsave.data(),psiM.data(),nels,nels);
    LogValue=InvertWithLog(psiM.data(),nels,nels,work.data(),LWork,pivot.data(),phase);
    copy_n(psiM.data(),nels*nels,psiMinv.data());

    if(omp_get_num_threads()==1)
    {
      checkIdentity(psiMsave,psiM,"Psi_0 * psiM(double)");
      checkIdentity(psiMsave,psiMinv,"Psi_0 * psiMinv(T)");
      checkDiff(psiM,psiMinv,"psiM(double)-psiMinv(T)");
    }
    */
  }

  ///recompute the inverse
  inline void recompute()
  {
    /*
    const int nels=psiV.size();
    INVT phase;
    transpose(psiMsave.data(),psiM.data(),nels,nels);
    InvertOnly(psiM.data(),nels,nels,work.data(),pivot.data(),LWork);
    copy_n(psiM.data(),nels*nels,psiMinv.data());
    */
  }

  /** return determinant ratio for the row replacement
   * @param iel the row (active particle) index
   */
  inline double ratio(int iel)
  {
    /*
    const int nels=psiV.size();
    constexpr T shift(0.5);
    constexpr INVT czero(0);
    for(int j=0; j<nels; ++j) psiV[j]=myRandom()-shift;
    curRatio=inner_product_n(psiV.data(),psiMinv[iel],nels,czero);
    return curRatio;
    */
  }

  /** accept the row and update the inverse */
  inline void accept(int iel)
  {
    /*
    const int nels=psiV.size();
    inverseRowUpdate(psiMinv.data(),psiV.data(),nels,nels,iel,curRatio);
    copy_n(psiV.data(),nels,psiMsave[iel]);
    */
  }
};

}

#endif
