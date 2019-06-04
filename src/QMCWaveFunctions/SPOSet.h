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
#include <Numerics/OhmmsPETE/OhmmsMatrix.h>

namespace qmcplusplus
{

class ParticleSet;

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
  using GradVector_t = Vector<GradType>;
  using ValueMatrix_t = Matrix<ValueType>;
  using GradMatrix_t = Matrix<GradType>;

  /// return the size of the orbital set
  inline int size() const { return OrbitalSetSize; }

  /// destructor
  virtual ~SPOSet() {}

  /// operates on a single walker
  /// evaluating SPOs
  virtual void evaluate_v(const PosType& p)   = 0;
  virtual void evaluate_vgl(const PosType& p) = 0;
  virtual void evaluate_vgh(const PosType& p) = 0;

  /** evaluate the values of this single-particle orbital set
   * @param P current ParticleSet
   * @param iat active particle
   * @param psi values of the SPO
   */
  virtual void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi)
  {
    //FIXME
  }

  /** evaluate the values, gradients and laplacians of this single-particle orbital set
   * @param P current ParticleSet
   * @param iat active particle
   * @param psi values of the SPO
   * @param dpsi gradients of the SPO
   * @param d2psi laplacians of the SPO
   */
  virtual void evaluate(const ParticleSet& P,
                        int iat,
                        ValueVector_t& psi,
                        GradVector_t& dpsi,
                        ValueVector_t& d2psi)
  {
    //FIXME
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
    //FIXME
  }

  /// operates on multiple walkers
  virtual void
      multi_evaluate_v(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list)
  {
    #pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate_v(pos_list[iw]);
  }

  virtual void
      multi_evaluate_vgl(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list)
  {
    #pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate_vgl(pos_list[iw]);
  }

  virtual void
      multi_evaluate_vgh(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list)
  {
    #pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate_vgh(pos_list[iw]);
  }
};

} // namespace qmcplusplus
#endif
