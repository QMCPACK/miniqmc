// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_H
#define QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_H

#include <string>
#include "Utilities/Configuration.h"
#include "Numerics/OhmmsPETE/OhmmsMatrix.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "CPU/SIMD/simd.hpp"

namespace qmcplusplus
{

/** base class for Single-particle orbital sets
 *
 * SPOSet stands for S(ingle)P(article)O(rbital)Set which contains
 * a number of single-particle orbitals with capabilities of evaluating \f$ \phi_j({\bf r}_i)\f$
 */
class SPOSet : public QMCTraits
{
protected:
  /// number of SPOs
  int OrbitalSetSize;
  /// name of the basis set
  std::string className;

public:
  using ValueVector_t = Vector<ValueType>;
  using GradVector_t  = Vector<GradType>;
  using ValueMatrix_t = Matrix<ValueType>;
  using GradMatrix_t  = Matrix<GradType>;

  /// return the size of the orbital set
  inline int size() const { return OrbitalSetSize; }

  /// destructor
  virtual ~SPOSet() {}

  /// operates on a single walker
  /** evaluate the values of this single-particle orbital set
   * @param P current ParticleSet
   * @param iat active particle
   * @param psi values of the SPO
   */
  virtual void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v) = 0;

  /** evaluate the values, gradients and laplacians of this single-particle orbital set
   * @param P current ParticleSet
   * @param iat active particle
   * @param psi values of the SPO
   * @param dpsi gradients of the SPO
   * @param d2psi laplacians of the SPO
   */
  virtual void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v, GradVector_t& dpsi_v, ValueVector_t& d2psi_v) = 0;

  /** evaluate determinant ratios for virtual moves, e.g., sphere move for nonlocalPP
   * @param VP virtual particle set
   * @param psi values of the SPO, used as a scratch space if needed
   * @param psiinv the row of inverse slater matrix corresponding to the particle moved virtually
   * @param ratios return determinant ratios
   */
  virtual void evaluateDetRatios(const VirtualParticleSet& VP,
                                 ValueVector_t& psi,
                                 const ValueVector_t& psiinv,
                                 std::vector<ValueType>& ratios)
  {
    assert(psi.size() == psiinv.size());
    for (int iat = 0; iat < VP.getTotalNum(); ++iat)
    {
      evaluate(VP, iat, psi);
      ratios[iat] = simd::dot(psi.data(), psiinv.data(), psi.size());
    }
  }

  /** evaluate the values, gradients and laplacians of this single-particle orbital for [first,last) particles
   * @param P current ParticleSet
   * @param first starting index of the particles
   * @param last ending index of the particles
   * @param logdet determinant matrix to be inverted
   * @param dlogdet gradients
   * @param d2logdet laplacians
   *
   */
  virtual void evaluate_notranspose(const ParticleSet& P,
                                    int first,
                                    int last,
                                    ValueMatrix_t& logdet,
                                    GradMatrix_t& dlogdet,
                                    ValueMatrix_t& d2logdet)
  {
    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
      ValueVector_t v(logdet[i], OrbitalSetSize);
      GradVector_t g(dlogdet[i], OrbitalSetSize);
      ValueVector_t l(d2logdet[i], OrbitalSetSize);
      evaluate(P, iat, v, g, l);
    }
  }

  /// operates on multiple walkers
  virtual void multi_evaluate(const std::vector<SPOSet*>& spo_list, const std::vector<ParticleSet*>& P_list, int iat,
                              const std::vector<ValueVector_t*>& psi_v_list)
  {
#pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate(*P_list[iw], iat, *psi_v_list[iw]);
  }

  virtual void multi_evaluate(const std::vector<SPOSet*>& spo_list, const std::vector<ParticleSet*>& P_list, int iat,
                              const std::vector<ValueVector_t*>& psi_v_list,
                              const std::vector<GradVector_t*>& dpsi_v_list,
                              const std::vector<ValueVector_t*>& d2psi_v_list)
  {
#pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate(*P_list[iw], iat, *psi_v_list[iw], *dpsi_v_list[iw], *d2psi_v_list[iw]);
  }

};

} // namespace qmcplusplus
#endif
