#ifndef QMCPLUSPLUS_TWOBODYJASTROW_KOKKOS_H
#define QMCPLUSPLUS_TWOBODYJASTROW_KOKKOS_H
#include <Kokkos_Core.hpp>
#include "Particle/ParticleSetKokkos.h"

namespace qmcplusplus
{

template<typename RealType, typename valT, int dim>
class TwoBodyJastrowKokkos{
public:
  Kokkos::View<valT[1]> LogValue;
  Kokkos::View<int[1]> Nelec;
  Kokkos::View<int[1]> NumGroups;
  Kokkos::View<int[2]> first; // starting index of each species
  Kokkos::View<int[2]> last; // one past ending index of each species
  Kokkos::View<int[1]> updateMode; // zero for ORB_PBYP_RATIO, one for ORB_PBYP_ALL,
                                   // two for ORB_PBYP_PARTIAL, three for ORB_WALKER

  Kokkos::View<valT[1]> cur_Uat; // temporary
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
  Kokkos::View<RealType*> cutoff_radius; // numGroups*numGroups elements
  Kokkos::View<RealType*> DeltaRInv; // numGroups*numGroups elements
  Kokkos::View<RealType**> SplineCoefs; // NumGroups*NumGroups x SplineCoefs
  Kokkos::View<RealType[16]> A, dA, d2A; // all 16 elements long, used for functor eval

  // default constructor
  KOKKOS_INLINE_FUNCTION
  TwoBodyJastrowKokkos() { ; }
  // Don't do a real constructor, just put this in TwoBodyJastrow

  // operator=
  // see if we can just say = default but keep the KOKKOS_INLINE_FUNCTION label
  KOKKOS_INLINE_FUNCTION
  TwoBodyJastrowKokkos& operator=(const TwoBodyJastrowKokkos& rhs) = default;

  TwoBodyJastrowKokkos(const TwoBodyJastrowKokkos&) = default;

  template<typename policyType, typename pskType>
  KOKKOS_INLINE_FUNCTION
  void acceptMove(policyType& pol, pskType& psk, int iel) {
    computeU3(pol, psk, iel, Kokkos::subview(psk.LikeDTDistances, iel, Kokkos::ALL()), old_u, old_du, old_d2u);
    if (updateMode(0) == 0)
    { // ratio-only during the move; need to compute derivatives
      computeU3(pol, psk, iel, psk.LikeDTTemp_r, cur_u, cur_du, cur_d2u);
    }

    valT cur_d2Uat(0);
    auto& new_dr = psk.LikeDTTemp_dr;
    auto old_dr = Kokkos::subview(psk.LikeDTDisplacements,iel,Kokkos::ALL(), Kokkos::ALL());
    constexpr valT lapfac = OHMMS_DIM - RealType(1);

    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(pol,Nelec(0)),
			    [&](const int& jel, valT& cursum) {
			      const valT du   = cur_u(jel) - old_u(jel);
			      const valT newl = cur_d2u(jel) + lapfac * cur_du(jel);
			      const valT dl   = old_d2u(jel) + lapfac * old_du(jel) - newl;
			      Uat(jel) += du;
			      d2Uat(jel) += dl;
			      cursum -= newl;
			    },cur_d2Uat);

    Kokkos::Array<valT,3> cur_dUat;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(pol,dim),
			 [&](const int& idim) {
			   valT cur_g  = cur_dUat[idim];
			   for (int jel = 0; jel < Nelec(0); jel++)
			   {
			     const valT newg     = cur_du(jel) * new_dr(jel,idim);
			     const valT dg       = newg - old_du(jel) * old_dr(jel,idim);
			     dUat(iel,idim)     -= dg;
			     cur_g              += newg;
			   }
			   cur_dUat[idim] = cur_g;
			 });

    LogValue(0) += Uat(iel) - cur_Uat(0);
    Uat(iel)     = cur_Uat(0);
    for (int idim = 0; idim < dim; idim++) {
      dUat(iel,idim)  = cur_dUat[idim];
    }
    d2Uat(iel) = cur_d2Uat;
  }

  template<typename policyType, typename pskType>
  KOKKOS_INLINE_FUNCTION
  valT ratio(policyType& pol, pskType& psk, int iel) {
    // only ratio, ready to compute it again
    updateMode(0) = 0; // ORB_PBYP_RATIO
    cur_Uat(0) = computeU(pol, psk, iel, psk.LikeDTTemp_r);
    return std::exp(Uat(iel) - cur_Uat(0));
  }

  template<typename policyType, typename pskType, typename gradType>
  KOKKOS_INLINE_FUNCTION
  valT ratioGrad(policyType& pol, pskType& psk, int iel, gradType inG) {
    updateMode(0) = 2; // ORB_PBYP_PARTIAL
    
    computeU3(pol, psk, iel, psk.LikeDTTemp_r, cur_u, cur_du, cur_d2u);

    Kokkos::single(Kokkos::PerTeam(pol),
		   [=] () {
		     cur_Uat(0) = 0.0;
		     for (int i = 0; i < Nelec(0); i++) {
		       cur_Uat(0) += cur_u(i);
		     }
		   });
    
    valT DiffVal = Uat(iel) - cur_Uat(0);

    Kokkos::Array<valT,3> tempG;
    accumulateG(pol, cur_du, Kokkos::subview(psk.LikeDTDisplacements, iel, Kokkos::ALL(), Kokkos::ALL()), tempG);
    for (int i = 0; i < dim; i++) {
      inG(i) += tempG[i];
    }
    return std::exp(DiffVal);
  }
      
  template<typename policyType, typename pskType>
  KOKKOS_INLINE_FUNCTION
  valT evaluateLog(policyType& pol, pskType& psk) {
    evaluateGL(pol, psk, true);
    return LogValue(0);
  }



  ////////////////////////////////////////////////////////////////////
  // helpers
  ////////////////////////////////////////////////////////////////////
  template<typename policyType, typename displType, typename gType>
  KOKKOS_INLINE_FUNCTION
  void accumulateG(policyType& pol, Kokkos::View<valT*> du, displType displ, gType grad) {
    for (int idim = 0; idim < dim; idim++) {
      grad[idim] = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(pol, Nelec(0)),
			      [=](const int& jat, valT& locVal) {
				locVal += du(jat) * displ(jat,idim);
			      },grad[idim]);
    }
  }

  template<typename policyType, typename pskType>
  KOKKOS_INLINE_FUNCTION
  void evaluateGL(policyType& pol, pskType& psk, bool fromscratch = false) {
    if (fromscratch) {
      recompute(pol, psk);
    }
    LogValue(0) = valT(0);
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(pol,Nelec(0)),
			    [=](const int& iel, valT& locVal) {
			      locVal += Uat(iel);
			    },LogValue(0));

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(pol,Nelec(0)),
			 [=](const int& iel) {
			   for (int d = 0; d < dim; d++) {
			     psk.G(iel,d) += dUat(iel,d);
			   }
			   psk.L(iel) += d2Uat(iel);
			 });
    constexpr valT mhalf(-0.5);
    LogValue(0) = mhalf * LogValue(0);
  }


  template<typename policyType, typename pskType>
  KOKKOS_INLINE_FUNCTION
  void recompute(policyType& pol, pskType& psk) {
    for (int ig = 0; ig < NumGroups(0); ++ig)
    {
      const int igt = ig * NumGroups(0);
      int last = psk.last(ig);
      
      Kokkos::parallel_for(Kokkos::TeamThreadRange(pol,last),
			   [&](const int& i) {
			     const int iel = psk.first(ig);
			     computeU3(pol, psk, iel, Kokkos::subview(psk.LikeDTDistances, iel, Kokkos::ALL()),
				       cur_u, cur_du, cur_d2u, true);
			     Uat(iel) = 0.0;
			     for (int j = 0; j < iel; j++) {
			       Uat(iel) += cur_u(j);
			     }

			     Kokkos::Array<RealType,3> grad;
			     valT lap(0);
			     
			     constexpr valT lapfac     = dim - RealType(1);
			     for (int jel = 0; jel < iel; ++jel)
			       lap += cur_d2u(jel) + lapfac * cur_du(jel);
			     for (int idim = 0; idim < dim; ++idim)
			     {
			       valT s                  = valT();
			       for (int jel = 0; jel < iel; ++jel)
				 s += cur_du(jel) * psk.LikeDTDisplacements(iel,jel,idim);
			       grad[idim] = s;
			     }
			     for(int idim = 0; idim < dim; idim++) {
			       dUat(iel,idim)  = grad[idim];
			     }
			     d2Uat(iel) = -lap;
			     // add the contribution from the upper triangle
			     for (int jel = 0; jel < iel; jel++)
			     {
			       Uat(jel) += cur_u(jel);
			       d2Uat(jel) -= cur_d2u(jel) + lapfac * cur_du[jel];
			     }
			     for (int idim = 0; idim < dim; ++idim)
			     {
			       for (int jel = 0; jel < iel; jel++)
				 dUat(jel,idim) -= cur_du(jel) * psk.LikeDTDisplacements(iel,jel,idim); 
			     }
			   });
    }
  }

  template<typename policyType, typename pskType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  valT computeU(policyType& pol, pskType& psk, int iel, distViewType dist) {
    valT curUat(0);
    const int igt = psk.GroupID(iel) * NumGroups(0);
    for (int jg = 0; jg < NumGroups(0); jg++) {
      const int istart = psk.first(jg);
      const int iend = psk.last(jg);
      curUat += FevaluateV(pol, jg, iel, istart, iend, dist);
    }
    return curUat;
  }

  template<typename policyType, typename pskType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void computeU3(policyType& pol, pskType& psk, int iel, distViewType dist, 
		 Kokkos::View<valT*> u, Kokkos::View<valT*> du, 
		 Kokkos::View<valT*> d2u, bool triangle = false) {
    const int jelmax = triangle ? iel : Nelec(0);
    constexpr valT czero(0);
    
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(pol,jelmax),
			 [&](const int& i) {
			   u(i) = czero;
			   du(i) = czero;
			   d2u(i) = czero;
			 });

    const int igt = psk.GroupID(iel) * NumGroups(0);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(pol, NumGroups(0)),
			 [&](const int& jg) {
			   const int istart = psk.first(jg);
			   int iend = jelmax;
			   if (psk.last(jg) < jelmax) {
			     iend = psk.last(jg);
			   }
			   FevaluateVGL(pol, igt+jg, iel, istart, iend, dist, u, du, d2u);
			 });
  }




  // all the functions to do evaluations for the 1D bspline functor come after here
  KOKKOS_INLINE_FUNCTION
  RealType Fevaluate(int gid, RealType r) {
    if (r >= cutoff_radius(gid))
      return 0.0;
    r *= DeltaRInv(gid);
    RealType ipart, t;
    t     = std::modf(r, &ipart);
    int i = (int)ipart;
    RealType tp[4];
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

  // update U, dU, d2U for a single electron
  KOKKOS_INLINE_FUNCTION
  void Fevaluate(int gid, int iel, RealType r) {
    if (r >= cutoff_radius(gid)) {
      Uat(iel) = 0.0;
      dUat(iel) = 0.0;
      d2Uat(iel) = 0.0;
    }
    r *= DeltaRInv(gid);
    const int i = (int)r;
    const RealType t = r - RealType(i); 

    RealType tp[4];
    tp[0] = t * t * t;
    tp[1] = t * t;
    tp[2] = t;
    tp[3] = 1.0;

    d2Uat(iel) = DeltaRInv(gid) * DeltaRInv(gid) *
      (SplineCoefs(gid,i+0)*(d2A( 0)*tp[0] + d2A( 1)*tp[1] + d2A( 2)*tp[2] + d2A( 3)*tp[3])+
       SplineCoefs(gid,i+1)*(d2A( 4)*tp[0] + d2A( 5)*tp[1] + d2A( 6)*tp[2] + d2A( 7)*tp[3])+
       SplineCoefs(gid,i+2)*(d2A( 8)*tp[0] + d2A( 9)*tp[1] + d2A(10)*tp[2] + d2A(11)*tp[3])+
       SplineCoefs(gid,i+3)*(d2A(12)*tp[0] + d2A(13)*tp[1] + d2A(14)*tp[2] + d2A(15)*tp[3]));
    dUat(iel) = DeltaRInv(gid) *
      (SplineCoefs(gid,i+0)*(dA( 0)*tp[0] + dA( 1)*tp[1] + dA( 2)*tp[2] + dA( 3)*tp[3])+
       SplineCoefs(gid,i+1)*(dA( 4)*tp[0] + dA( 5)*tp[1] + dA( 6)*tp[2] + dA( 7)*tp[3])+
       SplineCoefs(gid,i+2)*(dA( 8)*tp[0] + dA( 9)*tp[1] + dA(10)*tp[2] + dA(11)*tp[3])+
       SplineCoefs(gid,i+3)*(dA(12)*tp[0] + dA(13)*tp[1] + dA(14)*tp[2] + dA(15)*tp[3]));
    Uat(iel) =
      (SplineCoefs(gid,i+0)*(A( 0)*tp[0] + A( 1)*tp[1] + A( 2)*tp[2] + A( 3)*tp[3])+
       SplineCoefs(gid,i+1)*(A( 4)*tp[0] + A( 5)*tp[1] + A( 6)*tp[2] + A( 7)*tp[3])+
       SplineCoefs(gid,i+2)*(A( 8)*tp[0] + A( 9)*tp[1] + A(10)*tp[2] + A(11)*tp[3])+
       SplineCoefs(gid,i+3)*(A(12)*tp[0] + A(13)*tp[1] + A(14)*tp[2] + A(15)*tp[3]));
  }


  template<typename policyType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  RealType FevaluateV(policyType& pol, int gid, int iel, int start, int end, distViewType dist) {
    int iCount = 0;
    int iLimit = end - start;
    
    for (int jel = 0; jel < iLimit; jel++) {
      RealType r = dist(jel+start);
      if (r < cutoff_radius(0) && start + jel != iel) {
	DistCompressed(iCount) = r;
	iCount++;
      }
    }

    RealType d = 0.0;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(pol, iCount),
			    [&](const int& jel, RealType& locSum) {
			      RealType r = DistCompressed(jel);
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
			      locSum += (d1 + d2 + d3 + d4);
			    },d);
    return d;
  }
   
  template<typename policyType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void FevaluateVGL(policyType& pol, int gid, int iel, int start, int end, distViewType dist,
		    Kokkos::View<valT*> u, Kokkos::View<valT*> du, Kokkos::View<valT*> d2u) {
    RealType dSquareDeltaRinv = DeltaRInv(gid) * DeltaRInv(gid);
    constexpr RealType cOne(1);
    
    int iCount = 0;
    int iLimit = end - start;
    
    for (int jel = 0; jel < iLimit; jel++) {
      RealType r = dist(jel+start);
      if (r < cutoff_radius(gid) && start + jel != iel) {
	DistIndices(iCount) = jel+start;
	DistCompressed(iCount) = r;
	iCount++;
      }
    }
    
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(pol, iCount),
			 [&](const int& j) {
			   const RealType r = DistCompressed(j)*DeltaRInv(gid);
			   const RealType rinv = cOne / DistCompressed(j);
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
			   
			   d2u(iScatter) = dSquareDeltaRinv *
			     (sCoef0*( d2A( 2)*tp2 + d2A( 3))+
			      sCoef1*( d2A( 6)*tp2 + d2A( 7))+
			      sCoef2*( d2A(10)*tp2 + d2A(11))+
			      sCoef3*( d2A(14)*tp2 + d2A(15)));
			   
			   du(iScatter) = DeltaRInv(gid) * rinv *
			     (sCoef0*( dA( 1)*tp1 + dA( 2)*tp2 + dA( 3))+
			      sCoef1*( dA( 5)*tp1 + dA( 6)*tp2 + dA( 7))+
			      sCoef2*( dA( 9)*tp1 + dA(10)*tp2 + dA(11))+
			      sCoef3*( dA(13)*tp1 + dA(14)*tp2 + dA(15)));
      
			   u(iScatter) = (sCoef0*(A( 0)*tp0 + A( 1)*tp1 + A( 2)*tp2 + A( 3))+
					  sCoef1*(A( 4)*tp0 + A( 5)*tp1 + A( 6)*tp2 + A( 7))+
					  sCoef2*(A( 8)*tp0 + A( 9)*tp1 + A(10)*tp2 + A(11))+
					  sCoef3*(A(12)*tp0 + A(13)*tp1 + A(14)*tp2 + A(15)));
			 });
  }

};

}

#endif
