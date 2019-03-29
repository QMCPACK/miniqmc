#ifndef QMCPLUSPLUS_ONEBODY_JAS_DATA_H
#define QMCPLUSPLUS_ONEBODY_JAS_DATA_H
#include <Kokkos_Core.hpp>

namespace qmcplusplus
{

template<typename RealType>
class OneBodyJastrowData {
public:
  Kokkos::View<RealType*> LogValue; // one element
  Kokkos::View<int*> Nelec; // one element
  Kokkos::View<int*> Nions; // one element
  Kokkos::View<int*> NumGroups; // one element

  Kokkos::View<RealType*> curGrad; // OHMMS_DIM elements
  Kokkos::View<RealType*> curLap; // one element
  Kokkos::View<RealType*> curAt; // one elemnet

  Kokkos::View<RealType**> Grad; // Nelec x OHMMS_DIM
  Kokkos::View<RealType*> Lap; // Nelec long
  Kokkos::View<RealType*> Vat; // Nelec long
  Kokkos::View<RealType*> U, dU, d2U; // from base class, Nions long
  Kokkos::View<RealType*> DistCompressed; // Nions
  Kokkos::View<int*> DistIndices; // Nions

  // stuff for the one dimensional functors
  // should eventually add indirection so we can have different kinds of 
  // functors.  For the time being, we are even forcing them all to have
  // the same number of coefficients
  Kokkos::View<RealType*> cutoff_radius; // numGroups elements
  Kokkos::View<RealType*> DeltaRInv; // numGroups elements
  Kokkos::View<RealType**> SplineCoefs; // NumGroups x SplineCoefs
  Kokkos::View<RealType*> A, dA, d2A; // all 16 elements long, used for functor eval

  // default constructor
  KOKKOS_INLINE_FUNCTION
  OneBodyJastrowData() { ; }
  // Don't do a real constructor, just put this in OneBodyJastrow
  
  // operator=
  // see if we can just say = default but keep the KOKKOS_INLINE_FUNCTION label
  KOKKOS_INLINE_FUNCTION
  OneBodyJastrowData* operator=(const OneBodyJastrowData& rhs) {  
    LogValue = rhs.LogValue;
    Nelec = rhs.Nelec;
    Nions = rhs.Nions;
    NumGroups = rhs.NumGroups;
    curGrad = rhs.curGrad;
    curLap = rhs.curLap;
    curAt = rhs.curAt;
    Grad = rhs.Grad;
    Lap = rhs.Lap;
    Vat = rhs.Vat;
    U = rhs.U;
    dU = rhs.dU;
    d2U = rhs.d2U;
    DistCompressed = rhs.DistCompressed;
    DistIndices = rhs.DistIndices;
    cutoff_radius = rhs.cutoff_radius;
    DeltaRInv = rhs.DeltaRInv;
    SplineCoefs = rhs.SplineCoefs;
    A = rhs.A;
    dA = rhs.dA;
    d2A = rhs.d2A;
  }
  OneBodyJastrowData(const OneBodyJastrowData&) = default;
};

}  
    




#endif
