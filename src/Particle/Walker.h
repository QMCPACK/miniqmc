////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// D. Das, University of Illinois at Urbana-Champaign 
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Miguel Morales, moralessilva2@llnl.gov,
//    Lawrence Livermore National Laboratory
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Raymond Clay III, j.k.rofling@gmail.com,
//    Lawrence Livermore National Laboratory
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory 
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National
// Laboratory Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WALKER_H
#define QMCPLUSPLUS_WALKER_H

#include "OhmmsPETE/OhmmsMatrix.h"
#include "Utilities/PooledData.h"
#include <assert.h>
#include <deque>
namespace qmcplusplus
{

/** an enum denoting index of physical properties
 *
 * LOCALPOTENTIAL should be always the last enumeation
 * When a new enum is needed, modify ParticleSet::initPropertyList to match the
 * list
 */
enum
{
  LOGPSI = 0,      /*!< log(std::abs(psi)) instead of square of the many-body
                      wavefunction \f$|\Psi|^2\f$ */
  SIGN,            /*!< value of the many-body wavefunction \f$\Psi(\{R\})\f$ */
  UMBRELLAWEIGHT,  /*!< sum of wavefunction ratios for multiple H and Psi */
  R2ACCEPTED,      /*!< r^2 for accepted moves */
  R2PROPOSED,      /*!< r^2 for proposed moves */
  DRIFTSCALE,      /*!< scaling value for the drift */
  ALTERNATEENERGY, /*!< alternatelocal energy, the sum of all the components */
  LOCALENERGY,     /*!< local energy, the sum of all the components */
  LOCALPOTENTIAL, /*!< local potential energy = local energy - kinetic energy */
  NUMPROPERTIES   /*!< the number of properties */
};

/** A container class to represent a walker.
 *
 * A walker stores the particle configurations {R}  and a property container.
 * RealTypehe template (P)articleSet(A)ttribute is a generic container  of
 * position types.
 * RealTypehe template (G)radient(A)ttribute is a generic container of gradients
 * types.
 * Data members for each walker
 * - ID : identity for a walker. default is 0.
 * - Age : generation after a move is accepted.
 * - Weight : weight to take the ensemble averages
 * - Multiplicity : multiplicity for branching. Probably can be removed.
 * - Properties  : 2D container. RealTypehe first index corresponds to the H/Psi
 * index and second index >=NUMPROPERTIES.
 * - DataSet : anonymous container.
 */
template <typename t_traits, typename p_traits> struct Walker
{
  enum
  {
    DIM = t_traits::DIM
  };
  /** typedef for real data type */
  typedef typename t_traits::RealType RealType;
  /** typedef for estimator real data type */
  typedef typename t_traits::EstimatorRealType EstimatorRealType;
  /** typedef for value data type. */
  typedef typename t_traits::ValueType ValueType;
  /** array of particles */
  typedef typename p_traits::ParticlePos_t ParticlePos_t;
  /** array of gradients */
  typedef typename p_traits::ParticleGradient_t ParticleGradient_t;
  /** array of laplacians */
  typedef typename p_traits::ParticleLaplacian_t ParticleLaplacian_t;
  /** typedef for value data type. */
  typedef typename p_traits::ParticleValue_t ParticleValue_t;
  /// typedef for the property container, fixed size
  typedef Matrix<EstimatorRealType> PropertyContainer_t;
  /// typedef for buffer
  typedef PooledData<RealType> Buffer_t;

  /// id reserved for forward walking
  long ID;
  /// id reserved for forward walking
  long ParentID;
  /// DMCgeneration
  int Generation;
  /// Age of this walker age is incremented when a walker is not moved after a
  /// sweep
  int Age;
  /// Weight of the walker
  EstimatorRealType Weight;
  /** Number of copies for branching
   *
   * When Multiplicity = 0, this walker will be destroyed.
   */
  RealType Multiplicity;

  /** The configuration vector (3N-dimensional vector to store
     the positions of all the particles for a single walker)*/
  ParticlePos_t R;

  /// buffer for the data for particle-by-particle update
  Buffer_t DataSet;

  /// walker properties
  PropertyContainer_t Properties;

  /// create a walker for n-particles
  inline explicit Walker(int nptcl = 0)
  {
    ID           = 0;
    ParentID     = 0;
    Generation   = 0;
    Age          = 0;
    Weight       = 1.0;
    Multiplicity = 1.0;
    Properties.resize(1, NUMPROPERTIES);
    if (nptcl > 0) resize(nptcl);
  }

  inline Walker(const Walker &a) = default;
  inline ~Walker() {}

  /// assignment operator
  inline Walker &operator=(const Walker &a)
  {
    // make deep copy
    if (this != &a) makeCopy(a);
    return *this;
  }

  /// return the number of particles per walker
  inline int size() const { return R.size(); }

  /// resize for n particles
  inline void resize(int nptcl) { R.resize(nptcl); }

  /// copy the content of a walker
  inline void makeCopy(const Walker &a)
  {
    ID           = a.ID;
    ParentID     = a.ParentID;
    Generation   = a.Generation;
    Age          = a.Age;
    Weight       = a.Weight;
    Multiplicity = a.Multiplicity;
    if (R.size() != a.R.size()) resize(a.R.size());
    R       = a.R;
    DataSet = a.DataSet;
  }

  /** byte size for a packed message
   *
   * ID, Age, Properties, R, Drift, DataSet is packed
   */
  inline size_t byteSize()
  {
    size_t bsize = 0;
    return bsize;
  }

  template <class Msg> inline Msg &putMessage(Msg &m) { return m; }

  template <class Msg> inline Msg &getMessage(Msg &m) { return m; }
};

template <class RealType, class PA>
std::ostream &operator<<(std::ostream &out, const Walker<RealType, PA> &rhs)
{
  copy(rhs.Properties.begin(), rhs.Properties.end(),
       std::ostream_iterator<double>(out, " "));
  out << std::endl;
  out << rhs.R;
  return out;
}
}

#endif
