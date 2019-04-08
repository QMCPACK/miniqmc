#ifndef QMCPLUSPLUS_ONEBODY_JAS_DATA_H
#define QMCPLUSPLUS_ONEBODY_JAS_DATA_H
#include <Kokkos_Core.hpp>
#include "Particle/ParticleSetKokkos.h"

namespace qmcplusplus
{

template<typename RealType, int dim>
class OneBodyJastrowData {
public:
  using RealType == RealType;
  
  Kokkos::View<RealType[1]> LogValue; // one element
  Kokkos::View<int[1]> Nelec; // one element
  Kokkos::View<int[1]> Nions; // one element
  Kokkos::View<int[1]> NumGroups; // one element
  Kokkos::View<int[1]> updateMode; // zero for ORB_PBYP_RATIO, one for ORB_PBYP_ALL, two for ORB_PBYP_PARTIAL, three for ORB_WALKER

  Kokkos::View<RealType[dim]> curGrad; // OHMMS_DIM elements
  Kokkos::View<RealType[1]> curLap; // one element
  Kokkos::View<RealType[1]> curAt; // one elemnet

  Kokkos::View<RealType*[dim]> Grad; // Nelec x OHMMS_DIM
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
  Kokkos::View<RealType[16]> A, dA, d2A; // all 16 elements long, used for functor eval

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
    updateMode = rhs.updateMode;
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
  
  // 
  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  void acceptMove(pskType* psk, int iat) {
    if (updateMode(0) == 0) {
      computeU3(psk, psk->UnlikeDTTemp_r);
      curLap(0) = accumulateGL(P->UnlikeDTTemp_r, curGrad);
    }

    LogValue(0) += Vat(iat) - curAt(0);
    Vat(iat) = curAt(0);
    for (int i = 0; i < dims; i++) {
      Grad(iat,i) = curGrad(i);
    }
    Lap(iat) = curLap(0);
  }

  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  RealType ratio(pskType* psk, int iat) {
    updateMode(0) = 0;
    curAt(0) = computeU(psk, psk->UnlikeDTTemp_r);
    return std::exp(Vat(iat) - curAt);
  }

  // could later expand this to take a policy member and do hierarchical parallelism
  template<typename pskType, typename gradType>
  KOKKOS_INLINE_FUNCTION
  RealType ratioGrad(pskType* psk, int iat, gradType inG) {
    updateMode(0) = 2;
    computeU3(psk, psk->UnlikeDTTemp_r);
    curLap(0) = accumulateGL(P->UnlikeDTTemp_r, curGrad);
    curAt(0) = 0.0;
    for (int i = 0; i < Nions(0); i++) {
      curAt(0) += U(i);
    }
    for (int i = 0; i < dim; i++) {
      inG(i) = curGrad(i);
    }
    return std::exp(Vat[iat] - curAt);
  }

  // could later expand this to take a policy member and do hierarchical parallelism
  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  RealType evaluateLog(pskType* psk) {
    evaluateGL(psk, true);
    return LogValue(0);
  }

  // could later expand this to take a policy member and do hierarchical parallelism
  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  void evaluateGL(pskType* psk, bool fromscratch = false) {
    if (fromscratch) {
      recompute(psk);
    }

    for (size_t iel = 0; iel < Nelec(0); iel++) {
      for (int d = 0; d < dim; d++) {
	psk->G(iel,d) += Grad(iel,d);
      }
      psk->L(iel) -= L(iel);
      LogValue(0) -= Vat(iel);
    }
  }

  // could later expand this to take a policy member and do hierarchical parallelism
  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  void recompute(pskType* psk) {
    for (int iel = 0; iel < Nelec(0); iel++) {
      computeU3(psk, Kokkos::subview(psk->UnlikeDTDistances,iel,Kokkos::All()));
      Vat(iel) = 0.0;
      for (int iat = 0; iat < Nions(0); iat++) {
	Vat(iel) += U(iat);
      }
      Lap(iel) = accumulateGL(Kokkos::subview(psk->UnlikeDTDisplacements,iel,Kokkos::All(),Kokkos::All()),
			      Kokkos::subview(Grad, iel, Kokkos::All()));
    }
  }


  template<typename pskType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  RealType computeU(pskType* psk, distViewType dist) {
    RealType curVat(0);
    if (psk->numIonGroups(0) > 0) {
      for (int jg = 0; jg < psk->numIonGroups(0); jg++) {
	curVat += FevaluateV(jg, psk->ionFirst(jg), psk->ionLast(jg), dist); // note don't need check for self as
      	                                                             // this is always for unlike species
      }
    } else {
      for (int iat = 0; iat < Nions(0); iat++) {
	int gid = psk->ionGroupID(iat);
	curVat += Fevaluate(gid, dist(iat)); // needs to update U(iat), dU(iat), d2U(iat)
      }
    }
    return curVat;
  }

  // at the end, this will have updated U, dU and d2U for all atoms
  // could again make this amenable to hierarchical parallelism later
  template<typename pskType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void computeU3(pskType* psk, distViewType dist) {
    if (psk->numIonGroups(0) > 0) {
      for (int iat = 0; iat < Nions(0); iat++) {
	U(iat) = 0.0;
	dU(iat) = 0.0;
	d2U(iat) = 0.0;
      }
      for (int jg = 0; jg < psk->numIonGroups(0); jg++) {
	FevaluateVGL(jg, psk->ionFirst(jg), psk->ionLast(jg), dist); // note don't need check for self as
	                                                            // this is always for unlike species
      }
    } else {
      for (int iat = 0; iat < Nions(0); iat++) {
	int gid = psk->ionGroupID(iat);
	Fevaluate(gid, iat, dist(iat)); // needs to update U(iat), dU(iat), d2U(iat)
      }
    }
  }

  // just evaluate the value of the function for a given distance r and return
  KOKKOS_INLINE_FUNCTION
  RealType Fevaluate(int gid, RealType r) {
    if (r >= cutoff_radius(gid))
      return 0.0;
    r *= DeltaRInv(gid);
    real_type ipart, t;
    t     = std::modf(r, &ipart);
    int i = (int)ipart;
    real_type tp[4];
    tp[0] = t * t * t;
    tp[1] = t * t;
    tp[2] = t;
    tp[3] = 1.0;
    // clang-format off
    return
      (SplineCoefs(gid,i+0)*(A( 0)*tp[0] + A( 1)*tp[1] + A( 2)*tp[2] + A( 3)*tp[3])+
       SplineCoefs(gid,i+1)*(A( 4)*tp[0] + A( 5)*tp[1] + A( 6)*tp[2] + A( 7)*tp[3])+
       SplineCoefs(gid,i+2)*(A( 8)*tp[0] + A( 9)*tp[1] + A(10)*tp[2] + A(11)*tp[3])+
       SplineCoefs(gid,i+3)*(A(12)*tp[0] + A(13)*tp[1] + A(14)*tp[2] + A(15)*tp[3]));
    // clang-format on
  }


  // update U, dU and d2U for a single atom
  KOKKOS_INLINE_FUNCTION
  void Fevaluate(int gid, int iat, RealType r) {
    if (r >= cutoff_radius(gid)) {
      U(iat) = 0.0;
      dU(iat) = 0.0;
      d2U(iat) = 0.0;
    }
    r *= DeltaRInv(gid);
    const int i = (int)r;
    const RealType t = r - RealType(i); 

    RealType tp[4];
    tp[0] = t * t * t;
    tp[1] = t * t;
    tp[2] = t;
    tp[3] = 1.0;

    d2U(iat) = DeltaRInv(gid) * DeltaRInv(gid) *
      (SplineCoefs(gid,i+0)*(d2A( 0)*tp[0] + d2A( 1)*tp[1] + d2A( 2)*tp[2] + d2A( 3)*tp[3])+
       SplineCoefs(gid,i+1)*(d2A( 4)*tp[0] + d2A( 5)*tp[1] + d2A( 6)*tp[2] + d2A( 7)*tp[3])+
       SplineCoefs(gid,i+2)*(d2A( 8)*tp[0] + d2A( 9)*tp[1] + d2A(10)*tp[2] + d2A(11)*tp[3])+
       SplineCoefs(gid,i+3)*(d2A(12)*tp[0] + d2A(13)*tp[1] + d2A(14)*tp[2] + d2A(15)*tp[3]));
    dU(iat) = DeltaRInv(gid) *
      (SplineCoefs(gid,i+0)*(dA( 0)*tp[0] + dA( 1)*tp[1] + dA( 2)*tp[2] + dA( 3)*tp[3])+
       SplineCoefs(gid,i+1)*(dA( 4)*tp[0] + dA( 5)*tp[1] + dA( 6)*tp[2] + dA( 7)*tp[3])+
       SplineCoefs(gid,i+2)*(dA( 8)*tp[0] + dA( 9)*tp[1] + dA(10)*tp[2] + dA(11)*tp[3])+
       SplineCoefs(gid,i+3)*(dA(12)*tp[0] + dA(13)*tp[1] + dA(14)*tp[2] + dA(15)*tp[3]));
    U(iat) =
      (SplineCoefs(gid,i+0)*(A( 0)*tp[0] + A( 1)*tp[1] + A( 2)*tp[2] + A( 3)*tp[3])+
       SplineCoefs(gid,i+1)*(A( 4)*tp[0] + A( 5)*tp[1] + A( 6)*tp[2] + A( 7)*tp[3])+
       SplineCoefs(gid,i+2)*(A( 8)*tp[0] + A( 9)*tp[1] + A(10)*tp[2] + A(11)*tp[3])+
       SplineCoefs(gid,i+3)*(A(12)*tp[0] + A(13)*tp[1] + A(14)*tp[2] + A(15)*tp[3]));
  }

  template<typename distViewType>
  KOKKOS_INLINE_FUNCTION
  RealType FevaluateV(int gid, int start, int end, distViewType dist) {
    int iCount = 0;
    int iLimit = end - start;
    
    for (int jat = 0; jat < iLimit; jat++) {
      Realtype r = dist(jat+start);
      if (r < cutoff_radius(gid)) {
	DistArrayCompressed(iCount) = r;
	iCount++;
      }
    }

    RealType d = 0.0;
    for (int jat = 0; jat < iCount; jat++)
    {
      RealType r = distArrayCompressed(jat);
      r *= DeltaRInv(gid);
      int i         = (int)r;
      RealType t   = r - RealType(i);
      RealType tp0 = t * t * t;
      RealType tp1 = t * t;
      RealType tp2 = t;

      RealType d1 = SplineCoefs(gid,i + 0) * (A(0) * tp0 + A(1) * tp1 + A(2) * tp2 + A(3));
      RealType d2 = SplineCoefs(gid,i + 1) * (A(4) * tp0 + A(5) * tp1 + A(6) * tp2 + A(7));
      RealType d3 = SplineCoefs(gid,i + 2) * (A(8) * tp0 + A(9) * tp1 + A(10) * tp2 + A(11));
      RealType d4 = SplineCoefs(gid,i + 3) * (A(12) * tp0 + A(13) * tp1 + A(14) * tp2 + A(15));
      d += (d1 + d2 + d3 + d4);
    }
    return d;
  }
    
  
  // update U, dU and d2U for all atoms in a group starting at start and going to end
  template<typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void FevaluateVGL(int gid, int start, int end, distViewType dist) {
    RealType dSquareDeltaRinv = DeltaRInv(gid) * DeltaRInv(gid);
    constexpr RealType cOne(1);

    int iCount = 0;
    int iLimit = end - start;
    
    for (int jat = 0; jat < iLimit; jat++) {
      Realtype r = dist(jat+start);
      if (r < cutoff_radius(gid)) {
	DistIndices(iCount) = jat+start;
	DistArrayCompressed(iCount) = r;
	iCount++;
      }
    }

    for (int j = 0; j < iCount; j++) {
      const RealType r = DistArrayCompressed(j)*DeltaRInv(gid);
      const RealType rinv = cOne / DistArrayCompressed(j);
      const int iScatter   = DistIndices(j);
      const int iGather    = (int) r;

      const RealType t = r - RealType(iGather);
      const RealType tp0 = t*t*t;
      const RealType tp1 = t*t;
      const RealType tp2 = t;

      const RealType sCoef0 = SplineCoefs(gid, iGather+0);
      const RealType sCoef1 = SplineCoefs(gid, iGather+1);
      const RealType sCoef2 = SplineCoefs(gid, iGather+2);
      const RealType sCoef3 = SplineCoefs(gid, iGather+3);

      d2U(iScatter) = dSquareDeltaRinv *
	(sCoef0*( d2A( 2)*tp2 + d2A( 3))+
	 sCoef1*( d2A( 6)*tp2 + d2A( 7))+
	 sCoef2*( d2A(10)*tp2 + d2A(11))+
	 sCoef3*( d2A(14)*tp2 + d2A(15)));
      
      dU(iScatter) = DeltaRInv * rinv *
	(sCoef0*( dA( 1)*tp1 + dA( 2)*tp2 + dA( 3))+
	 sCoef1*( dA( 5)*tp1 + dA( 6)*tp2 + dA( 7))+
	 sCoef2*( dA( 9)*tp1 + dA(10)*tp2 + dA(11))+
	 sCoef3*( dA(13)*tp1 + dA(14)*tp2 + dA(15)));
      
      U(iScatter) = (sCoef0*(A( 0)*tp0 + A( 1)*tp1 + A( 2)*tp2 + A( 3))+
		     sCoef1*(A( 4)*tp0 + A( 5)*tp1 + A( 6)*tp2 + A( 7))+
		     sCoef2*(A( 8)*tp0 + A( 9)*tp1 + A(10)*tp2 + A(11))+
		     sCoef3*(A(12)*tp0 + A(13)*tp1 + A(14)*tp2 + A(15)));
    }
  }

  template<typename displViewType, typename gradViewType>
  KOKKOS_INLINE_FUNCTION
  RealType accumulateGL(displViewType& displ, gradViewType& grad) {
    RealType lap(0);
    constexpr RealType lapfac = dim - RealType(1);
    
    for (int jat = 0; jat < Nions(0); jat++) {
      lap += d2U(jat) + lapfac * du(jat);
    }
    for (int idim = 0; idim < dim; idim++) {
      RealType s(0.0);
      for (int jat = 0; jat < Nions(0); jat++) {
	s += dU(jat) * displ(jat,idim);
      }
      grad(idim) = s;
    }
    return lap;
  }




};

}

#endif
