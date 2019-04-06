#ifndef QMCPLUSPLUS_TWOBODYJASTROW_KOKKOS_H
#define QMCPLUSPLUS_TWOBODYJASTROW_KOKKOS_H
#include <Kokkos_Core.hpp>
#include "Particle/ParticleSetKokkos.h"

namespace qmcplusplus
{

template<typename RealType, typename valT, int dim>
class TwoBodyJastrowKokkos{
public:
  Kokkos::View<RealType[1]> LogValue;
  Kokkos::View<int[1]> Nelec;
  Kokkos::View<int[1]> NumGroups;
  Kokkos::View<int[2]> first; // starting index of each species
  Kokkos::View<int[2]> last; // one past ending index of each species
  Kokkos::View<int[1]> updateMode; // zero for ORB_PBYP_RATIO, one for ORB_PBYP_ALL,
                                   // two for ORB_PBYP_PARTIAL, three for ORB_WALKER

  Kokkos::View<valT*> Uat; // nelec
  Kokkos::View<valT*[dim], Kokkos::LayoutLeft> dUat; // nelec
  Kokkos::View<valT*> d2Uat; // nelec
  
  Kokkos::View<valT*> cur_u, cur_du, cur_d2u; // nelec long
  Kokkos::View<valT*> old_u, old_du, old_d2u; // nelec long

  Kokkos::View<RealType*> DistCompressed; // Nions
  Kokkos::View<int*> DistIndices; // Nions

  // stuff for the one dimensional functors
  // should eventually add indirection so we can have different kinds of 
  // functors.  For the time being, we are even forcing them all to have
  // the same number of coefficients
  Kokkos::View<RealType*> cutoff_radius; // numGroups elements
  Kokkos::View<RealType*> DeltaRInv; // numGroups elements
  Kokkos::View<RealType**> SplineCoefs; // NumGroups x SplineCoefs
  Kokkos::View<RealType[16]> A, dA, d2A; // all 16 elements long, used for functor eval

  // default constructor
  KOKKOS_INLINE_FUNCTION
  TwoBodyJastrowKokkos() { ; }
  // Don't do a real constructor, just put this in TwoBodyJastrow

  // operator=
  // see if we can just say = default but keep the KOKKOS_INLINE_FUNCTION label
  KOKKOS_INLINE_FUNCTION
  TwoBodyJastrowKokkos* operator=(const TwoBodyJastrowKokkos& rhs) {  
    LogValue = rhs.LogValue;
    Nelec = rhs.Nelec;
    NumGroups = rhs.NumGroups;
    first = rhs.first;
    last = rhs.last;
    updateMode = rhs.updateMode;
    Uat = rhs.Uat;
    dUat = rhs.dUat;
    d2Uat = rhs.d2Uat;
    cur_u = rhs.cur_u;
    cur_du = rhs.cur_du;
    cur_d2u = rhs.cur_d2u;
    old_u = rhs.old_u;
    old_du = rhs.old_du;
    old_d2u = rhs.old_d2u;
    DistCompressed = rhs.DistCompressed;
    DistIndices = rhs.DistIndices;
    cutoff_radius = rhs.cutoff_radius;
    DeltaRInv = rhs.DeltaRInv;
    SplineCoefs = rhs.SplineCoefs;
    A = rhs.A;
    dA = rhs.dA;
    d2A = rhs.d2A;
  }

  TwoBodyJastrowKokkos(const TwoBodyJastrowKokkos&) = default;
};

}

#endif
