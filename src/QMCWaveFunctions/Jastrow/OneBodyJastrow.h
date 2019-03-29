//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_ONEBODYJASTROW_H
#define QMCPLUSPLUS_ONEBODYJASTROW_H
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include <Utilities/SIMD/allocator.hpp>
#include <Utilities/SIMD/algorithm.hpp>
#include <numeric>
#include "OneBodyJastrowData.h"

/*!
 * @file OneBodyJastrow.h
 */

namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for one-body Jastrow function using multiple functors
 */
template<class FT>
struct OneBodyJastrow : public WaveFunctionComponent
{
  using jasDataType = OneBodyJastrowData<FT::real_type>;
  jasDataType jasData;
  bool splCoefsNotAllocated;

#ifdef QMC_PARALLEL_JASTROW
  typedef Kokkos::TeamPolicy<> policy_t;
#else
  typedef Kokkos::TeamPolicy<Kokkos::Serial> policy_t;
#endif
  /// alias FuncType
  using FuncType = FT;
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using RowContainer = DistanceTableData::RowContainer;
  /// table index
  int myTableID;
  /// number of ions
  int Nions;
  /// number of electrons
  int Nelec;
  /// number of groups
  int NumGroups;
  /// reference to the sources (ions)
  const ParticleSet& Ions;

  valT curAt;
  valT curLap;
  posT curGrad;

  ///\f$Vat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Vat;
  Kokkos::View<valT*> U, dU, d2U;
  Kokkos::View<valT*> DistCompressed;
  Kokkos::View<int*> DistIndice;
  Vector<posT> Grad;
  Vector<valT> Lap;
  /// Container for \f$F[ig*NumGroups+jg]\f$
  typedef Kokkos::Device<
            Kokkos::DefaultHostExecutionSpace,
            typename Kokkos::DefaultExecutionSpace::memory_space>
       F_device_type;
  Kokkos::View<FT*,F_device_type> F;

  //Kokkos temporary arrays, a la two body jastrow.
  int iat, igt, jg_hack;
  const RealType* dist;
  int first[2], last[2];
  RealType*   u;
  RealType*  du;
  RealType* d2u;

  OneBodyJastrow(const ParticleSet& ions, ParticleSet& els) : Ions(ions)
  {
    initalize(els);
    myTableID                 = els.addTable(ions, DT_SOA);
    WaveFunctionComponentName = "OneBodyJastrow";
    InitializeJastrowData();
  }

  OneBodyJastrow(const OneBodyJastrow& rhs) = default;

  ~OneBodyJastrow()
  {
 //   for (int i = 0; i < F.size(); ++i)
 //     if (F[i] != nullptr)
 //       delete F[i];
  }

  /* initialize storage */
  void initalize(ParticleSet& els)
  {
    Nions     = Ions.getTotalNum();
    NumGroups = Ions.getSpeciesSet().getTotalNum();
    int fsize = std::max(NumGroups,4); //Odd choice.  Why 4?
                                       //Ignore for now and port.
                                       
    if (NumGroups > 1 && !Ions.IsGrouped)
    {
      NumGroups = 0;
    }
    Nelec = els.getTotalNum();
    Vat.resize(Nelec);
    Grad.resize(Nelec);
    Lap.resize(Nelec);

    U              = Kokkos::View<valT*>("U",Nions);
    dU             = Kokkos::View<valT*>("dU",Nions);
    d2U            = Kokkos::View<valT*>("d2U",Nions);
    DistCompressed = Kokkos::View<valT*>("DistCompressed",Nions);
    DistIndice     = Kokkos::View<int*>("DistIndice",Nions);

    F = Kokkos::View<FT*,F_device_type>("FT",std::max(NumGroups,4));
    for(int i=0; i<fsize; i++){
      new (&F(i)) FT();
    }
  }

  // note the way I am doing this, I am 
  void initializeJastrowData() {
    jasData.LogValue       = Kokkos::View<valT*>("LogValue", 1);

    jasData.Nelec          = Kokkos::View<int*>("Nelec", 1);
    auto NelecMirror       = Kokkos::create_mirror_view(jasData.Nelec);
    NelecMirror(0)         = Nelec;
    Kokkos::deep_copy(jasData.Nelec, NelecMirror);
    
    jasData.Nions          = Kokkos::View<int*>("Nions", 1);
    auto NionsMirror       = Kokkos::create_mirror_view(jasData.Nions);
    NionsMirror(0)         = Nions;
    Kokkos::deep_copy(jasData.Nions, NionsMirror);
    
    jasData.NumGroups      = Kokkos::View<int*>("NumGroups", 1);
    auto NumGroupsMirror   = Kokkos::create_mirror_view(jasData.NumGroups);
    NumGroupsMirror(0)     = NumGroups;
    Kokkos::deep_copy(jasData.NumGroups, NumGroupsMirror);
    
    // these things are just zero on the CPU, so don't have to set their values
    jasData.curGad         = Kokkos::View<valT*>("curGrad", OHMMS_DIM);
    jasData.Grad           = Kokkos::View<valT**>("Grad", Nelec, OHMMS_DIM);
    jasData.curLap         = Kokkos::View<valT*>("curLap", 1);
    jasData.Lap            = Kokkos::View<valT*>("Lap", Nelec);
    jasData.curAt          = Kokkos::View<valT*>("curAt", 1);
    jasData.Vat            = Kokkos::View<valT*>("Vat", Nelec);
    
    // these things are already views, so just do operator=
    jasData.U              = U;
    jasData.dU             = dU;
    jasData.d2U            = d2U;
    jasData.DistCompressed = DistCompressed;
    jasData.DistIndices    = DistIndice;
    
    // need to put in the data for A, dA and d2A    
    TinyVector<valT, 16> A(-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
			    3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
			   -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
			    1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0);
    TinyVector<valT,16>  dA(0.0, -0.5,  1.0, -0.5,
			    0.0,  1.5, -2.0,  0.0,
			    0.0, -1.5,  1.0,  0.5,
			    0.0,  0.5,  0.0,  0.0);
    TinyVector<ValT,16>  d2A(0.0, 0.0, -1.0,  1.0,
			     0.0, 0.0,  3.0, -2.0,
			     0.0, 0.0, -3.0,  1.0,
			     0.0, 0.0,  1.0,  0.0);

    jasData.A              = Kokkos::View<valT*>("A", 16);
    auto Amirror           = Kokkos::create_mirror_view(jasData.A);
    jasData.dA             = Kokkos::View<valT*>("dA", 16);
    auto dAmirror           = Kokkos::create_mirror_view(jasData.dA);
    jasData.d2A            = Kokkos::View<valT*>("d2A", 16);
    auto d2Amirror           = Kokkos::create_mirror_view(jasData.d2A);

    for (int i = 0; i < 16; i++) {
      Amirror(i) = A[i];
      dAmirror(i) = dA[i];
      d2Amirror(i) = d2A[i];
    }
    Kokkos::deep_copy(jasData.A, Amirror);
    Kokkos::deep_copy(jasData.dA, dAmirror);
    Kokkos::deep_copy(jasData.d2A, d2Amirror);


    // also set up and allocate memory for cutoff_radius, DeltaRInv
    jasData.cutoff_radius   = Kokkos::View<valT*>("Cutoff_Radii", NumGroups);
    jasData.DeltaRInv       = Kokkos::View<valT*>("DeltaRInv", NumGroups);

    // unfortunately have to defer setting up SplineCoefs because we don't yet know
    // how many elements are in SplineCoefs on the cpu
    splCoefsNotAllocated   = true;
  }

  void addFunc(int source_type, FT* afunc, int target_type = -1)
  {
    //if (F[source_type] != nullptr)
    //  delete F[source_type];
    F[source_type] = *afunc;

    // also copy in cutoff_radius, DeltaRInv
    auto crMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
    auto drinvMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
    Kokkos::deep_copy(crMirror, jasData.cutoff_radius);
    Kokkos::deep_copy(drinvMirror, jasData.DeltaRInv);
    crMirror(source_type) = afunc->cutoff_radius;
    drinvMirror(source_type) = afunc->DeltaRInv;
    Kokkos::deep_copy(jasData.cutoff_radius, crMirror);
    Kokkos::deep_copy(jasData.DeltaRInv, drinvMirror);

    //if necessary set up SplineCoefs view on device and then copy data it
    if (splCoefsNotAllocated) {
      splCoefsNotAllocated = false;
      jasData.SplineCoefs   = Kokkos::View<valT*>("SplineCoefficients", NumGroups, afunc->SplineCoefs.extent(0));
    }
    auto bigScMirror   = Kokkos::create_mirror_view(jasData.SplineCoefs);
    auto smallScMirror = Kokkos::create_mirror_view(afunc->SplineCoefs);
    Kokkos::deep_copy(smallScMirror, afunc->SplineCoefs);
    Kokkos::deep_copy(bigScMirror,   jasData.SplineCoefs);
    for (int i = 0; i < afunc->SplineCoefs.extent(0); i++) {
      bigScMirror(source_type, i) = smallScMirror(i);
    }
    Kokkos::deep_copy(jasData.SplineCoefs, bigScMirror);
  }

  template<aobjdType, aobjpsdType, vectorType, vectorType2>
  void populateCollectiveViews(aobjType aobjd, aobjpsdType aobjpsd, vectorType& WFC_list, vectorType2& P_list) {
    auto aobjdMirror = Kokkos::create_mirror_view(aobjd);
    auto aobjpsdMirror = Kokkos::create_mirror_view(aobjpsd);

    for (int i = 0; i < WFC_list.size(); i++) {
      aobjdMirror(i) = static_cast<OneBodyJastrow*>(WFC_list[i])->jasData;
      aobjpsdMirror(i) = P_list[i]->getOneBodyRelevantData();
    }
    Kokkos::deep_copy(aobjd, aobjdMirror);
    Kokkos::deep_copy(aobjpsd, aobjpsdMirror);
  }

  template<aobjdType, aobjpsdType, vectorType, vectorType2>
  void populateCollectiveViews(aobjType aobjd, aobjpsdType aobjpsd, vectorType& WFC_list, vectorType2& P_list, std::vector<bool>& isAccepted) {
    auto aobjdMirror = Kokkos::create_mirror_view(aobjd);
    auto aobjpsdMirror = Kokkos::create_mirror_view(aobjpsd);

    int idx = 0;
    for (int i = 0; i < WFC_list.size(); i++) {
      if (isAccepted[i]) {
	aobjdMirror(idx) = static_cast<OneBodyJastrow*>(WFC_list[i])->jasData;
	aobjpsdMirror(idx) = P_list[i]->getOneBodyRelevantData();
	idx++;
      }
    }
    Kokkos::deep_copy(aobjd, aobjdMirror);
    Kokkos::deep_copy(aobjpsd, aobjpsdMirror);
  }
  

  // note that G_list and L_list feel redundant, they are just elements of P_list
  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
                                 const std::vector<ParticleSet*>& P_list,
                                 const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
                                 const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
                                 ParticleSet::ParticleValue_t& values) {
    
    // make a view of all of the OneBodyJastrowData and relevantParticleSetData
    Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
    Kokkos::View<objParticleSetData*> allOneBodyJastrowParticleSetData("aobjpsd", P_list.size());
    populateCollectiveViews(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, WFC_list, P_list);

    // need to make a view to hold all of the output LogValues
    Kokkos::View<double*> tempValues("tempValues", P_list.size());

    // need to write this function
    doOneBodyJastrowMultiEvaluateLog(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, tempValues);

    // copy the results out to values
    auto tempValMirror = Kokkos::create_mirror_view(tempValues);
    Kokkos::deep_copy(tempValMirror, tempValues);
    
    for (int i = 0; i < P_list.size(); i++) {
      values[i] = tempValMirror(i);
    }
  }

  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			      const std::vector<ParticleSet*>& P_list,
                              int iat,
			      std::vector<posT>& grad_now) {

    // make a view of all of the OneBodyJastrowData and relevantParticleSetData
    Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
    Kokkos::View<objParticleSetData*> allOneBodyJastrowParticleSetData("aobjpsd", P_list.size());
    populateCollectiveViews(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, WFC_list, P_list);
    
    // need to make a view to hold all of the output LogValues
    Kokkos::View<double**> grad_now_view("tempValues", P_list.size(), OHMMS_DIM);

    // need to write this function
    doOneBodyJastrowMultiEvalGrad(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, grad_now_view);

    // copy the results out to values
    auto grad_now_view_mirror = Kokkos::create_mirror_view(grad_now_view);
    Kokkos::deep_copy(grad_now_view_mirror, grad_now_view);
    
    for (int i = 0; i < P_list.size(); i++) {
      for (int j = 0; j < OHMMS_DIM; j++) {
	grad_now[i][j] = grad_now_view_mirror(i,j);
      }
    }
  }

  
  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			       const std::vector<ParticleSet*>& P_list,
			       int iat,
			       std::vector<valT>& ratios,
			       std::vector<posT>& grad_new) {

    // make a view of all of the OneBodyJastrowData and relevantParticleSetData
    Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
    Kokkos::View<objParticleSetData*> allOneBodyJastrowParticleSetData("aobjpsd", P_list.size());
    populateCollectiveViews(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, WFC_list, P_list);
    
    // need to make a view to hold all of the output LogValues
    Kokkos::View<double**> grad_new_view("tempValues", P_list.size(), OHMMS_DIM);
    Kokkos::View<double*> ratios_view("ratios", P_list.size());
    
    // need to write this function
    doOneBodyJastrowMultiRatioGrad(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, grad_new_view, ratios_view);

    // copy the results out to values
    auto grad_new_view_mirror = Kokkos::create_mirror_view(grad_new_view);
    Kokkos::deep_copy(grad_new_view_mirror, grad_new_view);
    auto ratios_view_mirror = Kokkos::create_mirror_view(ratios_view);
    Kokkos::deep_copy(ratios_view_mirror, ratios_view);
    
    for (int i = 0; i < P_list.size(); i++) {
      ratios[i] = ratios_view_mirror(i);
      for (int j = 0; j < OHMMS_DIM; j++) {
	grad_now[i][j] = grad_now_view_mirror(i,j);
      }
    }
  }

  virtual void multi_acceptRestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
				       const std::vector<ParticleSet*>& P_list,
				       const std::vector<bool>& isAccepted,
				       int iat) {
    int numAccepted = 0;
    for (int i = 0; i < isAccepted.size(); i++) {
      if (isAccepted[i]) {
	numAccepted++;
      }
    }
    
    // make a view of all of the OneBodyJastrowData and relevantParticleSetData
    Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", numAccepted); 
    Kokkos::View<objParticleSetData*> allOneBodyJastrowParticleSetData("aobjpsd", numAccepted);
    populateCollectiveViews(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, WFC_list, P_list, isAccepted);
    
    // need to write this function
    doOneBodyJastrowMultiAcceptRestoreMove(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, iat);
    
    // be careful on this one, looks like it is being done for side effects.  Should see what needs to go back!!!
  }

  virtual void multi_evaluateGL(const std::vector<WaveFunctionComponent*>& WFC_list,
				const std::vector<ParticleSet*>& P_list,
				const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
				const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
				bool fromscratch = false) {
    
    // make a view of all of the OneBodyJastrowData and relevantParticleSetData
    Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
    Kokkos::View<objParticleSetData*> allOneBodyJastrowParticleSetData("aobjpsd", P_list.size());
    populateCollectiveViews(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, WFC_list, P_list);

    // need to write this function
    doOneBodyJastrowMultiEvaluateGL(allOneBodyJastrowData, allOneBodyJastrowParticleSetData, fromscratch);

    // know that we will need LogValue to up updated after this, possibly other things in ParticleSet!!!
    for (int i = 0; i < WFC_list.size(); i++) {
      auto LogValueMirror = Kokkos::create_mirror_view(static_cast<OneBodyJastrow*>(WFC_list[i])->OneBodyJastrowData.LogValue);
      Kokkos::deep_copy(LogValueMirror, static_cast<OneBodyJastrow*>(WFC_list[i])->OneBodyJastrowData.LogValue);
      LogValue = LogValueMirror(0);
    }
  }
  
  
  ///////////////////////// end internal multi functions

  void recompute(ParticleSet& P)
  {
    const DistanceTableData& d_ie(*(P.DistTables[myTableID]));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      computeU3(P, iat, d_ie.Distances[iat]);
      Vat[iat] = simd::accumulate_n(U.data(), Nions, valT());
      Lap[iat] = accumulateGL(dU.data(), d2U.data(), d_ie.Displacements[iat], Grad[iat]);
    }
  }

  RealType evaluateLog(ParticleSet& P,
                       ParticleSet::ParticleGradient_t& G,
                       ParticleSet::ParticleLaplacian_t& L)
  {
    evaluateGL(P, G, L, true);
    return LogValue;
  }

  ValueType ratio(ParticleSet& P, int iat)
  {
    UpdateMode = ORB_PBYP_RATIO;
    curAt      = computeU(P.DistTables[myTableID]->Temp_r.data());
    return std::exp(Vat[iat] - curAt);
  }

  inline valT computeU(const valT* dist)
  {
    valT curVat(0);
    if (NumGroups > 0)
    {
      for (int jg = 0; jg < NumGroups; ++jg)
      {
      //  if (F[jg] != nullptr)
          curVat += F[jg].evaluateV(-1, Ions.first(jg), Ions.last(jg), dist, DistCompressed.data());
      }
    }
    else
    {
      for (int c = 0; c < Nions; ++c)
      {
        int gid = Ions.GroupID[c];
     //   if (F[gid] != nullptr)
          curVat += F[gid].evaluate(dist[c]);
      }
    }
    return curVat;
  }

  inline void evaluateGL(ParticleSet& P,
                         ParticleSet::ParticleGradient_t& G,
                         ParticleSet::ParticleLaplacian_t& L,
                         bool fromscratch = false)
  {
    if (fromscratch)
      recompute(P);

    for (size_t iat = 0; iat < Nelec; ++iat)
      G[iat] += Grad[iat];
    for (size_t iat = 0; iat < Nelec; ++iat)
      L[iat] -= Lap[iat];
    LogValue = -simd::accumulate_n(Vat.data(), Nelec, valT());
  }

  /** compute gradient and lap
   * @return lap
   */
  inline valT accumulateGL(const valT* restrict du,
                           const valT* restrict d2u,
                           const RowContainer& displ,
                           posT& grad) const
  {
    valT lap(0);
    constexpr valT lapfac = OHMMS_DIM - RealType(1);
    for (int jat = 0; jat < Nions; ++jat)
      lap += d2u[jat] + lapfac * du[jat];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT* restrict dX = displ.data(idim);
      valT s                  = valT();
      for (int jat = 0; jat < Nions; ++jat)
        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
    return lap;
  }

  /** compute U, dU and d2U
   * @param P quantum particleset
   * @param iat the moving particle
   * @param dist starting address of the distances of the ions wrt the iat-th
   * particle
   */
  inline void computeU3(ParticleSet& P, int iat_, const valT* dist_)
  {
    if (NumGroups > 0)
    { 
      iat = iat_;
      dist = dist_;
      u = U.data();
      du = dU.data();
      d2u = d2U.data(); 
      // ions are grouped
      constexpr valT czero(0);
      std::fill_n(U.data(), Nions, czero);
      std::fill_n(dU.data(), Nions, czero);
      std::fill_n(d2U.data(), Nions, czero);

      for (int jg = 0; jg < NumGroups; ++jg)
      {
    /*    F[jg].evaluateVGL(-1,
                           Ions.first(jg),
                           Ions.last(jg),
                           dist,
                           U.data(),
                           dU.data(),
                           d2U.data(),
                           DistCompressed.data(),
                           DistIndice.data());*/
         first[jg] = Ions.first(jg);
         last[jg]  = Ions.last(jg);
         jg_hack=jg;
         Kokkos::parallel_for(policy_t(1,1,32),*this);
      }
    }
    else
    {
      for (int c = 0; c < Nions; ++c)
      {
        int gid = Ions.GroupID[c];
     //   if (F[gid] != nullptr)
        if (true)
        {
          U[c] = F[gid].evaluate(dist[c], dU[c], d2U[c]);
          dU[c] /= dist[c];
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator() (const typename policy_t::member_type& team) const{
    int jg = jg_hack;
    int iStart = first[jg];
    int iEnd   = last[jg];
    F[jg].evaluateVGL(team,
                      -1,
                      iStart,
                      iEnd,
                      dist,
                      u,
                      du,
                      d2u,
                      DistCompressed.data(),
                      DistIndice.data());

  } 

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   */
  GradType evalGrad(ParticleSet& P, int iat) { return GradType(Grad[iat]); }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   *
   * Using Temp_r. curAt, curGrad and curLap are computed.
   */
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    computeU3(P, iat, P.DistTables[myTableID]->Temp_r.data());
    curLap = accumulateGL(dU.data(), d2U.data(), P.DistTables[myTableID]->Temp_dr, curGrad);
    curAt  = simd::accumulate_n(U.data(), Nions, valT());
    grad_iat += curGrad;
    return std::exp(Vat[iat] - curAt);
  }

  /** Accpted move. Update Vat[iat],Grad[iat] and Lap[iat] */
  void acceptMove(ParticleSet& P, int iat)
  {
    if (UpdateMode == ORB_PBYP_RATIO)
    {
      computeU3(P, iat, P.DistTables[myTableID]->Temp_r.data());
      curLap = accumulateGL(dU.data(), d2U.data(), P.DistTables[myTableID]->Temp_dr, curGrad);
    }

    LogValue += Vat[iat] - curAt;
    Vat[iat]  = curAt;
    Grad[iat] = curGrad;
    Lap[iat]  = curLap;
  }
};

} // namespace qmcplusplus
#endif
