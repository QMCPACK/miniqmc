////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
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
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WAVEFUNCTIONCOMPONENTBASE_H
#define QMCPLUSPLUS_WAVEFUNCTIONCOMPONENTBASE_H
#include "Utilities/Configuration.h"
#include "Particle/ParticleSet.h"
#include "Particle/DistanceTableData.h"
#include "QMCWaveFunctions/WaveFunctionKokkos.h"

/**@file WaveFunctionComponent.h
 *@brief Declaration of WaveFunctionComponent
 */
namespace qmcplusplus
{
/// forward declaration of WaveFunctionComponent
class WaveFunctionComponent;

typedef WaveFunctionComponent* WaveFunctionComponentPtr;

/**@defgroup WaveFunctionComponent Wavefunction Component group
 * @brief Classes which constitute a many-body trial wave function
 *
 * A many-body trial wave function is
 * \f[
 \Psi(\{ {\bf R}\}) = \prod_i \psi_{i}(\{ {\bf R}\}),
 * \f]
 * where \f$\Psi\f$s are represented by
 * the derived classes from WaveFunctionComponent.
 */
/** @ingroup WaveFunctionComponentComponent
 * @brief An abstract class for a component of a many-body trial wave function
 */
struct WaveFunctionComponent : public QMCTraits
{
  /// recasting enum of DistanceTableData to maintain consistency
  enum
  {
    SourceIndex  = DistanceTableData::SourceIndex,
    VisitorIndex = DistanceTableData::VisitorIndex,
    WalkerIndex  = DistanceTableData::WalkerIndex
  };

  /** enum for a update mode */
  enum
  {
    ORB_PBYP_RATIO,   /*!< particle-by-particle ratio only */
    ORB_PBYP_ALL,     /*!< particle-by-particle, update Value-Gradient-Laplacian */
    ORB_PBYP_PARTIAL, /*!< particle-by-particle, update Value and Grdient */
    ORB_WALKER,       /*!< walker update */
    ORB_ALLWALKER     /*!< all walkers update */
  };

  typedef ParticleAttrib<ValueType> ValueVectorType;
  typedef ParticleAttrib<GradType> GradVectorType;
  typedef PooledData<RealType> BufferType;
  typedef ParticleSet::Walker_t Walker_t;

  /** flag to set the optimization mode */
  bool IsOptimizing;
  /** boolean to set optimization
   *
   * If true, this object is actively modified during optimization
   */
  bool Optimizable;
  /** current update mode */
  int UpdateMode;
  /** current \f$\log\phi \f$
   */
  RealType LogValue;
  /** current phase
   */
  RealType PhaseValue;
  /** A vector for \f$ \frac{\partial \nabla \log\phi}{\partial \alpha} \f$
   */
  GradVectorType dLogPsi;
  /** A vector for \f$ \frac{\partial \nabla^2 \log\phi}{\partial \alpha} \f$
   */
  ValueVectorType d2LogPsi;
  /** Name of this wavefunction component
   */
  std::string WaveFunctionComponentName;

  /// default constructor
  WaveFunctionComponent()
      : IsOptimizing(false),
        Optimizable(true),
        UpdateMode(ORB_WALKER),
        LogValue(0.0),
        PhaseValue(0.0),
        WaveFunctionComponentName("WaveFunctionComponent")
  {}

  /// default destructor
  virtual ~WaveFunctionComponent() {}

  /// operates on multiple walkers
  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
				 WaveFunctionKokkos& wfc,
				 Kokkos::View<ParticleSet::pskType*>& psk,
				 ParticleSet::ParticleValue_t& values) {
    //
  };

  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			      WaveFunctionKokkos& wfc,
			      Kokkos::View<ParticleSet::pskType*> psk,
			      int iat,
			      std::vector<PosType>& grad_now)
  {
    //
  };

  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			       WaveFunctionKokkos& wfc,
			       Kokkos::View<ParticleSet::pskType*> psk,
			       int iel,
			       Kokkos::View<int*>& isValidMap, int numValid,
			       std::vector<ValueType>& ratios,
			       std::vector<PosType>& grad_new)
  {
    //
  };

  virtual void multi_acceptrestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
				       WaveFunctionKokkos& wfc,
				       Kokkos::View<ParticleSet::pskType*> psk,
				       Kokkos::View<int*>& isAcceptedMap, int numAccepted, int iel)
  {
    //
  };

  virtual void multi_ratio(const std::vector<WaveFunctionComponent*>& WFC_list,
                           const std::vector<ParticleSet*>& P_list,
                           int iat,
                           ParticleSet::ParticleValue_t& ratio_list){
      // TODO
  };

  virtual void multi_evalRatio(int pairNum, Kokkos::View<int***>& eiList,
			       WaveFunctionKokkos& wfc,
			       Kokkos::View<ParticleSetKokkos<RealType, ValueType, 3>*>& apsk,
			       Kokkos::View<RealType***>& likeTempR,
			       Kokkos::View<RealType***>& unlikeTempR,
			       std::vector<ValueType>& ratios) {
    // TODO
  };

  virtual void multi_evaluateGL(WaveFunctionKokkos& wfc,
				Kokkos::View<ParticleSetKokkos<RealType, ValueType, 3>*>& apsk,
				bool fromscratch = false) {
    //
  }

}; 

}// namespace qmcplusplus
#endif
