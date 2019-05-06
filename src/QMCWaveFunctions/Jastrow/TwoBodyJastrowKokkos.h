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
  Kokkos::View<valT[1]> temporaryScratch; 
  Kokkos::View<valT[dim]> temporaryScratchDim; 

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

    // Number of electrons Reduction
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(pol,Nelec(0)),
			    [&](const int& jel, valT& cursum) {
			      const valT du   = cur_u(jel) - old_u(jel);
			      const valT newl = cur_d2u(jel) + lapfac * cur_du(jel);
			      const valT dl   = old_d2u(jel) + lapfac * old_du(jel) - newl;
			      Uat(jel) += du;
			      d2Uat(jel) += dl;
			      cursum -= newl;
			    },cur_d2Uat);

    Kokkos::Array<valT,3> cur_dUat;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(pol,dim),
			 [&](const int& idim) {
			   valT cur_g  = cur_dUat[idim];
			   // Number of Electrons Reduction
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

  // this does the two calls to computeU3
  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  void acceptMove_part1(pskType& psk, int iel, int workingElNum) {
    computeU3(psk, iel, Kokkos::subview(psk.LikeDTDistances, iel, Kokkos::ALL()), 
	      old_u, old_du, old_d2u, workingElNum);
    if (updateMode(0) == 0) {
      computeU3(psk, iel, psk.LikeDTTemp_r, cur_u, cur_du, cur_d2u, workingElNum);
    }
    if (workingElNum == 0) {
      temporaryScratch(0) = 0.0;
      for (int i = 0; i < dim; i++) {
	//temporaryScratchDim(i) = cur_dUat(i);
	temporaryScratchDim(i) = 0.0;
      }
    }
  }

  // now everything is set-up from computeU3
  // this is to be the main part of a reduction
  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  void acceptMove_part2(pskType& psk, int iel, int workingElNum) {
    valT lapfac = OHMMS_DIM - RealType(1);
    const valT du   = cur_u(workingElNum) - old_u(workingElNum);
    const valT newl = cur_d2u(workingElNum) + lapfac * cur_du(workingElNum);
    const valT dl   = old_d2u(workingElNum) + lapfac * old_du(workingElNum) - newl;
    Uat(workingElNum) += du;
    d2Uat(workingElNum) += dl;
    // the sum of -newl for all electrons goes into a temporary and then eventually into d2Uat(iel)
    Kokkos::atomic_add(&temporaryScratch(0),-newl);

    for (int d = 0; d < dim; d++) {
      const valT newg = cur_du(workingElNum) * psk.LikeDTTemp_dr(workingElNum,d);
      const valT dg   = newg - old_du(workingElNum) * psk.LikeDTDisplacements(iel,workingElNum,d);
      Kokkos::atomic_add(&(dUat(iel,d)),-dg);
      // eventually this is bound for d2Uat(iel,d)
      Kokkos::atomic_add(&(temporaryScratchDim(d)),newg);
    }
  } 

  KOKKOS_INLINE_FUNCTION
  void acceptMove_part3(int iel) {
    for (int d = 0; d < dim; d++) {
      dUat(iel,d) = temporaryScratchDim(d);
    }
    LogValue(0) += Uat(iel) = cur_Uat(0);
    Uat(iel) = cur_Uat(0);
    d2Uat(iel) = temporaryScratch(0);
  }

  // so, every thing coming in will get an electron number and a walker number separately
  // seem s like this should not be too bad for computeU3

  // when passing into this, psk is now on the host via UVM
  template<typename pskType>
  void acceptMove(Kokkos::DefaultExecutionSpace pol, pskType& psk, int iel) {

    // this is bad too.  I'm having to go throuh UVM for stuff like the psk.LikeDTDistances
    // and also old_u etc.  At least these are just views, so the amount of data transferred is small
    computeU3(pol, psk, iel, Kokkos::subview(psk.LikeDTDistances, iel, Kokkos::ALL()), old_u, old_du, old_d2u);
    //
    if (updateMode(0) == 0)
    { // ratio-only during the move; need to compute derivatives
      computeU3(pol, psk, iel, psk.LikeDTTemp_r, cur_u, cur_du, cur_d2u);
    }

    valT cur_d2Uat(0);
    // again. more. UVM
    auto new_dr = psk.LikeDTTemp_dr;
    auto old_dr = Kokkos::subview(psk.LikeDTDisplacements,iel,Kokkos::ALL(), Kokkos::ALL());
    constexpr valT lapfac = OHMMS_DIM - RealType(1);

    // need to do this because we're a member function.  Could probably fix by making bare functions
    // and passing in the appropriate twoBodyJastrow as well
    auto me = *this;

    // Number of electrons Reduction
    auto result = Kokkos::subview(me.d2Uat,iel); // result is now a view that has a single element

    // very good thing about this is that we are doing a RangePolicy<> so we can get the whole device
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(pol,0,Nelec(0)),
        KOKKOS_LAMBDA (const int& jel, valT& cursum) {
            const valT du   = me.cur_u(jel) - me.old_u(jel);
            const valT newl = me.cur_d2u(jel) + lapfac * me.cur_du(jel);
            const valT dl   = me.old_d2u(jel) + lapfac * me.old_du(jel) - newl;
            me.Uat(jel) += du;
            me.d2Uat(jel) += dl;
            cursum -= newl;
          },result);
    Kokkos::fence();

    Kokkos::Array<valT,3> cur_dUat;
    for(int idim = 0; idim<dim; idim++) {

      // again putting place we reduce into into a 1d subview
      // also getting another reduction out of the Kokkos::atomic_add
      // probably better to think about a reduciton into a struct or something like that
      auto result2 = Kokkos::subview(me.dUat,iel,idim);
      valT cur_g  = cur_dUat[idim];
      // Number of Electrons Reduction
      Kokkos::parallel_reduce(Kokkos::RangePolicy<>(pol,0,Nelec(0)),
			      KOKKOS_LAMBDA (const int& jel, valT& cursum) {
				const valT newg     = me.cur_du(jel) * new_dr(jel,idim);
				const valT dg       = newg - me.old_du(jel) * old_dr(jel,idim);
				Kokkos::atomic_add(&(me.dUat(iel,idim)),-dg);
				cursum              += newg;
			      },result2);
      Kokkos::fence();
      //cur_dUat[idim] = cur_g;
    }

    // degenerate parallel for, just so this will execute on the device,
    // even if it is slow
    Kokkos::parallel_for(1,KOKKOS_LAMBDA (const int) {
      me.LogValue(0) += me.Uat(iel) - me.cur_Uat(0);
      me.Uat(iel)     = me.cur_Uat(0);
    });
  }

  template<typename policyType, typename pskType>
  KOKKOS_INLINE_FUNCTION
  valT ratio(policyType& pol, pskType& psk, int iel) {
    // only ratio, ready to compute it again
    updateMode(0) = 0; // ORB_PBYP_RATIO
    cur_Uat(0) = computeU(pol, psk, iel, psk.LikeDTTemp_r);
    return std::exp(Uat(iel) - cur_Uat(0));
  }

  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  void ratioGrad_part1(pskType& psk, int iel, int workingElNum) {
    updateMode(0) = 2;
    computeU3(psk, iel, psk.LikeDTTemp_r, cur_u, cur_du, cur_d2u, workingElNum);
    if (workingElNum == 0) {
      cur_Uat(0) = 0.0;
      for (int i = 0; i < dim; i++) {
	temporaryScratchDim(i) = 0.0;
      }
    }
  }

  // do reduction for cur_Uat, probably better to use standard reduction
  // but it is a bookkeping issue.  Maybe look later after timing
  template<typename pskType>
  KOKKOS_INLINE_FUNCTION
  void ratioGrad_part2(pskType& psk, int iel, int workingElNum) {
    Kokkos::atomic_add(&(cur_Uat(0)), cur_u(workingElNum));
    for (int i = 0; i < dim; i++) {
      Kokkos::atomic_add(&(temporaryScratchDim(i)), cur_du(workingElNum) * psk.LikeDTDisplacements(iel, workingElNum, i));
    }
  }

  template<typename gradType>
  KOKKOS_INLINE_FUNCTION
  valT ratioGrad_part3(int iel, gradType& inG) {
    valT DiffVal = Uat(iel) - cur_Uat(0);
    for (int i = 0; i < dim; i++) {
      inG(i) += temporaryScratchDim(i);
    }
    return std::exp(DiffVal);
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
    int iel = pol.league_rank()%Nelec(0);
    LogValue(0) = valT(0);
    Kokkos::single(Kokkos::PerTeam(pol),[&]() {
      Kokkos::atomic_add(&LogValue(0),0.5*Uat(iel));
      for (int d = 0; d < dim; d++) {
        psk.G(iel,d) += dUat(iel,d);
      }
      psk.L(iel) += d2Uat(iel);
    });
    /*
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(pol,Nelec(0)),
			    [=](const int& iel, valT& locVal) {
			      locVal += Uat(iel);
			    },LogValue(0));

    Kokkos::parallel_for(Kokkos::TeamVectorRange(pol,Nelec(0)),
			 [=](const int& iel) {
			   for (int d = 0; d < dim; d++) {
			     psk.G(iel,d) += dUat(iel,d);
			   }
			   psk.L(iel) += d2Uat(iel);
			 });
    */
  }


  template<typename policyType, typename pskType>
  KOKKOS_INLINE_FUNCTION
  void recompute(policyType& pol, pskType& psk) {
    // 2
    for (int ig = 0; ig < NumGroups(0); ++ig)
    {
      const int igt = ig * NumGroups(0);
      int count = psk.last(ig)-psk.first(ig);
      
      const int i = pol.league_rank()%Nelec(0);
      // number of electrons/2

      //Kokkos::parallel_for(Kokkos::TeamThreadRange(pol,last),
			//   [&](const int& i) {
      if(i<count) {
			     const int iel = psk.first(ig)+i;
			     computeU3(pol, psk, iel, Kokkos::subview(psk.LikeDTDistances, iel, Kokkos::ALL()),
				       cur_u, cur_du, cur_d2u, true);
			     Kokkos::single(Kokkos::PerTeam(pol),[&] (){
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
			     pol.team_barrier();
			   }
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

  // this is the newest one
  template<typename pskType, typename distViewType, typename devRatioType>
  KOKKOS_INLINE_FUNCTION
  void computeU(pskType& psk, int iel, distViewType dist, int workingElNum, devRatioType& devRatios, int walkerIndex, int knotNum) {
    const int igt = psk.GroupID(iel) * NumGroups(0);
    const int workingElGroup = psk.GroupID(workingElNum);
    const int functorGroupID = igt+workingElGroup;
    RealType val = FevaluateV(functorGroupID, iel, dist, workingElNum);
    Kokkos::atomic_add(&(devRatios(walkerIndex, knotNum)), val);
  }


  template<typename policyType, typename pskType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void computeU3(policyType& pol, pskType& psk, int iel, distViewType dist, 
		 Kokkos::View<valT*> u, Kokkos::View<valT*> du, 
		 Kokkos::View<valT*> d2u, bool triangle = false) {
    const int jelmax = triangle ? iel : Nelec(0);
    constexpr valT czero(0);
    
    Kokkos::parallel_for(Kokkos::TeamVectorRange(pol,jelmax),
			 [&](const int& i) {
			   u(i) = czero;
			   du(i) = czero;
			   d2u(i) = czero;
			 });

    const int igt = psk.GroupID(iel) * NumGroups(0);
    for(int jg = 0; jg<NumGroups(0); jg++) {
      const int istart = psk.first(jg);
      int iend = jelmax;
      if (psk.last(jg) < jelmax) {
	iend = psk.last(jg);
      }
      FevaluateVGL(pol, igt+jg, iel, istart, iend, dist, u, du, d2u);
    }
  }
  
  // this is the newest one that is only doing work for a single workingElectron
  template<typename pskType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void computeU3(pskType& psk, int iel, distViewType dist,
		   Kokkos::View<valT*> u, Kokkos::View<valT*> du,
		   Kokkos::View<valT*> d2u, int workingElNum, bool triangle = false) {
    const int jelmax = triangle ? iel : Nelec(0);
    if (workingElNum < jelmax) {
      u(workingElNum) = valT(0);
      du(workingElNum) = valT(0);
      d2u(workingElNum) = valT(0);
    }
    const int igt = psk.GroupID(iel) * NumGroups(0);
    const int workingElGroup = psk.GroupID(workingElNum);
    const int functorGroupID = igt+workingElGroup;
    FevaluateVGL(functorGroupID, iel, dist, u, du, d2u, workingElNum);
  }


  template<typename pskType, typename distViewType>
  void computeU3(Kokkos::DefaultExecutionSpace& pol, pskType& psk, int iel, distViewType dist,
		 Kokkos::View<valT*> u, Kokkos::View<valT*> du,
		 Kokkos::View<valT*> d2u, bool triangle = false) {
    const int jelmax = triangle ? iel : Nelec(0);
    constexpr valT czero(0);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(pol,0,jelmax),
       KOKKOS_LAMBDA (const int& i) {
         u(i) = czero;
         du(i) = czero;
         d2u(i) = czero;
       });

    const int igt = psk.GroupID(iel) * NumGroups(0);
    for(int jg = 0; jg<NumGroups(0); jg++)   {
      const int istart = psk.first(jg);
      int iend = jelmax;
      if (psk.last(jg) < jelmax) {
	iend = psk.last(jg);
      }
      FevaluateVGL(pol, igt+jg, iel, istart, iend, dist, u, du, d2u);
    }
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


  template<typename distViewType>
  KOKKOS_INLINE_FUNCTION
  RealType FevaluateV(int gid, int iel, distViewType& dist, int workingElNum) {
    RealType d = 0.0;
    if (dist(workingElNum) < cutoff_radius(gid) && iel != workingElNum) {
      const RealType r = dist(workingElNum) * DeltaRInv(gid);
      const int i = (int)r;
      const RealType t = r - RealType(i);
      const RealType tp0 = t*t*t;
      const RealType tp1 = t*t;
      const RealType tp2 = t;
      
      const RealType d1 = SplineCoefs(gid,i + 0) * (A(0) * tp0 + A(1) * tp1 + A(2) * tp2 + A(3));
      const RealType d2 = SplineCoefs(gid,i + 1) * (A(4) * tp0 + A(5) * tp1 + A(6) * tp2 + A(7));
      const RealType d3 = SplineCoefs(gid,i + 2) * (A(8) * tp0 + A(9) * tp1 + A(10) * tp2 + A(11));
      const RealType d4 = SplineCoefs(gid,i + 3) * (A(12) * tp0 + A(13) * tp1 + A(14) * tp2 + A(15));
      d = d1+d2+d3+d4;
    }
    return d;
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
  
  template<typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void FevaluateVGL(int gid, int iel, distViewType& dist, Kokkos::View<valT*>& u,
		    Kokkos::View<valT*>& du, Kokkos::View<valT*> d2u, int workingElNum) {

    if (dist(workingElNum) < cutoff_radius(gid) && workingElNum != iel) {
      const RealType dSquareDeltaRinv = DeltaRInv(gid) * DeltaRInv(gid);
      const RealType cOne(1);
      const RealType r = dist(workingElNum) * DeltaRInv(gid);
      const RealType rinv = cOne / dist(workingElNum);
      const int iGather = (int) r;
      
      const RealType t = r - RealType(iGather);
      const RealType tp0 = t*t*t;
      const RealType tp1 = t*t;
      const RealType tp2 = t;
      
      const RealType sCoef0 = SplineCoefs(gid, iGather+0);
      const RealType sCoef1 = SplineCoefs(gid, iGather+1);
      const RealType sCoef2 = SplineCoefs(gid, iGather+2);
      const RealType sCoef3 = SplineCoefs(gid, iGather+3);

      d2u(workingElNum) = dSquareDeltaRinv *
	(sCoef0*( d2A( 2)*tp2 + d2A( 3))+
	 sCoef1*( d2A( 6)*tp2 + d2A( 7))+
	 sCoef2*( d2A(10)*tp2 + d2A(11))+
	 sCoef3*( d2A(14)*tp2 + d2A(15)));
      
      du(workingElNum) = DeltaRInv(gid) * rinv *
	(sCoef0*( dA( 1)*tp1 + dA( 2)*tp2 + dA( 3))+
	 sCoef1*( dA( 5)*tp1 + dA( 6)*tp2 + dA( 7))+
	 sCoef2*( dA( 9)*tp1 + dA(10)*tp2 + dA(11))+
	 sCoef3*( dA(13)*tp1 + dA(14)*tp2 + dA(15)));
      
      u(workingElNum) = (sCoef0*(A( 0)*tp0 + A( 1)*tp1 + A( 2)*tp2 + A( 3))+
			 sCoef1*(A( 4)*tp0 + A( 5)*tp1 + A( 6)*tp2 + A( 7))+
			 sCoef2*(A( 8)*tp0 + A( 9)*tp1 + A(10)*tp2 + A(11))+
			 sCoef3*(A(12)*tp0 + A(13)*tp1 + A(14)*tp2 + A(15)));
    }
  }
  

  template<typename policyType, typename distViewType>
  KOKKOS_INLINE_FUNCTION
  void FevaluateVGL(policyType& pol, int gid, int iel, int start, int end, distViewType dist,
		    Kokkos::View<valT*> u, Kokkos::View<valT*> du, Kokkos::View<valT*> d2u) {
    RealType dSquareDeltaRinv = DeltaRInv(gid) * DeltaRInv(gid);
    constexpr RealType cOne(1);
    
    int iCount = 0;
    int iLimit = end - start;
    if(pol.team_rank()==0) {
    Kokkos::parallel_scan(Kokkos::ThreadVectorRange(pol,iLimit),
			  [&] (const int jel, int& mycount, bool final) {
			    RealType r = dist(jel+start);
			    if (r < cutoff_radius(gid) && start + jel != iel) {
			      if(final) {
				DistIndices(mycount) = jel+start;
				DistCompressed(mycount) = r;
			      }
			      mycount++;
			    }
			  });
    }
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(pol,iLimit),
        [&] (const int jel, int& mycount) {
      RealType r = dist(jel+start);
      if (r < cutoff_radius(gid) && start + jel != iel) {
        mycount++;
      }
    },iCount);
    pol.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(pol, iCount),
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

  template<typename distViewType>
  void FevaluateVGL(Kokkos::DefaultExecutionSpace& pol, int gid, int iel, int start, int end, distViewType dist,
        Kokkos::View<valT*> u, Kokkos::View<valT*> du, Kokkos::View<valT*> d2u) {
    RealType dSquareDeltaRinv = DeltaRInv(gid) * DeltaRInv(gid);
    constexpr RealType cOne(1);

    int iCount = 0;
    int iLimit = end - start;
    //Kokkos::fence();
    auto me = *this;
    //printf("%i %i %i %i %i %s %s %p\n",gid,iel,start,dist.extent(0),cutoff_radius.extent(0),dist.label().c_str(),cutoff_radius.label().c_str(),cutoff_radius.data());


    Kokkos::parallel_scan("FevaluateVGL::par_scan",Kokkos::RangePolicy<>(0,iLimit),
        KOKKOS_LAMBDA(const int jel, int& mycount, bool final) {

      RealType r = dist(jel+start);
      if (r < me.cutoff_radius(gid) && start + jel != iel) {
        if(final) {
          me.DistIndices(mycount) = jel+start;
          me.DistCompressed(mycount) = r;
        }
        mycount++;
      }
    },iCount);

    Kokkos::parallel_for("FevaluateVGL::par_for",Kokkos::RangePolicy<>(pol, 0,iCount),
        KOKKOS_LAMBDA  (const int& j) {
         const RealType r = me.DistCompressed(j)*me.DeltaRInv(gid);
         const RealType rinv = cOne / me.DistCompressed(j);
         const int iScatter   = me.DistIndices(j);
         const int iGather    = (int) r;

         const RealType t = r - RealType(iGather);
         const RealType tp0 = t*t*t;
         const RealType tp1 = t*t;
         const RealType tp2 = t;

         const RealType sCoef0 = me.SplineCoefs(gid, iGather+0);
         const RealType sCoef1 = me.SplineCoefs(gid, iGather+1);
         const RealType sCoef2 = me.SplineCoefs(gid, iGather+2);
         const RealType sCoef3 = me.SplineCoefs(gid, iGather+3);

         d2u(iScatter) = dSquareDeltaRinv *
           (sCoef0*( me.d2A( 2)*tp2 + me.d2A( 3))+
            sCoef1*( me.d2A( 6)*tp2 + me.d2A( 7))+
            sCoef2*( me.d2A(10)*tp2 + me.d2A(11))+
            sCoef3*( me.d2A(14)*tp2 + me.d2A(15)));

         du(iScatter) = me.DeltaRInv(gid) * rinv *
           (sCoef0*( me.dA( 1)*tp1 + me.dA( 2)*tp2 + me.dA( 3))+
            sCoef1*( me.dA( 5)*tp1 + me.dA( 6)*tp2 + me.dA( 7))+
            sCoef2*( me.dA( 9)*tp1 + me.dA(10)*tp2 + me.dA(11))+
            sCoef3*( me.dA(13)*tp1 + me.dA(14)*tp2 + me.dA(15)));

         u(iScatter) = (sCoef0*(me.A( 0)*tp0 + me.A( 1)*tp1 + me.A( 2)*tp2 + me.A( 3))+
            sCoef1*(me.A( 4)*tp0 + me.A( 5)*tp1 + me.A( 6)*tp2 + me.A( 7))+
            sCoef2*(me.A( 8)*tp0 + me.A( 9)*tp1 + me.A(10)*tp2 + me.A(11))+
            sCoef3*(me.A(12)*tp0 + me.A(13)*tp1 + me.A(14)*tp2 + me.A(15)));
       });
  }

};

}

#endif
