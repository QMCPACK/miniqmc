#ifndef PARTICLE_SET_KOKKOS_H
#define PARTICLE_SET_KOKKOS_H
#include <Kokkos_Core.hpp>

namespace qmcplusplus
{

// note that some functionality provided by Lattice will be folded into this
// toUnit, outOfBound, isValid, WignerSeitzRadius?,
// toUnit_floor (whatever is needed to enable DTD_BConds)

  
// for the moment, don't think I really need:  PCID, mass, Z, LRBox
//                                            UseSphereUpdate, SameMass, newRedPos
template<typename RealType, typename ValueType, int dim>
class ParticleSetKokkos
{
public:
  Kokkos::View<int*>                                      ID; // nparticles long, unique int for each particle
  Kokkos::View<int*>                                      IndirectID; // nparticles long (if IsGrouped=true, ID = IndirectID)   
  Kokkos::View<int*>                                      GroupID; // nparticles long, label to say which species
  Kokkos::View<int*>                                      SubPtcl; // number of groups+1 long.  Each element is the start of a group.  
                                                                   // Last element is one beyond the end
  Kokkos::View<RealType*[dim],Kokkos::LayoutRight>        R; // nparticles by dims  (dims dimension is the fast one)
  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft>         RSoA; // dims by nparticles  (particles dimension is fast one)
  Kokkos::View<ValueType*[dim],Kokkos::LayoutLeft>        G; // nparticles by dims to hold (possibly complex) gradients
  Kokkos::View<ValueType*>                                L; // nparticles long to hold (possibly complex) laplacians
  Kokkos::View<bool[1]>                                   UseBoundBox; // single boolean value
  Kokkos::View<bool[1]>                                   IsGrouped; // single boolean value
  Kokkos::View<int[1]>                                    activePtcl; // index pointing to the active particle for single particle moves
  Kokkos::View<RealType[dim]>                             activePos; // position of the active particle
  // See if SpeciesSet becomes necessary

  // Figure out what is necessary to carry out operations on DistTables that are called for
  // -- move, evaluate, moveOnSphere
  //   evaluate calls DTD_BConds::computeDistances,  uses R, RSoA, Distances, Displacements
  //   moveOnSphere also calls DTD_BConds::computeDistances, uses internal Temp_R and Temp_dr
  //   move uses moveOnSphere to do its work  (BA version uses Origin a lot)

  // for the moment, consider the possibility of only allowing two distance tables here
  // one would be like-like (AA) and the other would be like unlike (AB).  The problem I'm having
  // is that I don't know the dimension of the
  Kokkos::View<RealType[dim][dim]>                      DT_G; // [dim][dim]
  Kokkos::View<RealType[dim][dim]>                      DT_R; // [dim][dim]
  Kokkos::View<RealType[8][dim],Kokkos::LayoutLeft>     corners; // [8][dim]
  Kokkos::View<int[dim]>                                BoxBConds;
  
  Kokkos::View<RealType**>                              LikeDTDistances; // [nparticles][nparticles]
  Kokkos::View<RealType**[dim]>                         LikeDTDisplacements; // [nparticles][nparticles][dim]
  Kokkos::View<RealType*>                               LikeDTTemp_r; // [nparticles]
  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft>       LikeDTTemp_dr; // [nparticles][dim]

  Kokkos::View<RealType**>                              UnlikeDTDistances; // [nparticles][ntargets]
  Kokkos::View<RealType**[dim]>                         UnlikeDTDisplacements; // [nparticles][ntargets][dim]
  Kokkos::View<RealType*>                               UnlikeDTTemp_r; // [ntargets]
  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft>       UnlikeDTTemp_dr; // [ntargets][dim]

  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft>       originR; // [ntargets][dim]  // locations of all the particles in the source set
  // will probably need to include a groups array for the ions as well
  Kokkos::View<int[1]>                                  numIonGroups;
  Kokkos::View<int*>                                    ionGroupID;
  Kokkos::View<int*>                                    ionSubPtcl;

  
  // default constructor
  KOKKOS_INLINE_FUNCTION
  ParticleSetKokkos() { ; }

  KOKKOS_INLINE_FUNCTION
  ParticleSetKokkos& operator=(const ParticleSetKokkos& rhs) = default;

  ParticleSetKokkos(const ParticleSetKokkos&) = default;

  KOKKOS_INLINE_FUNCTION
  int first(int group) const { return SubPtcl(group); }

  KOKKOS_INLINE_FUNCTION
  int last(int group) const { return SubPtcl(group+1); }

  KOKKOS_INLINE_FUNCTION
  int ionFirst(int group) const { return ionSubPtcl(group); }

  KOKKOS_INLINE_FUNCTION
  int ionLast(int group) const { return ionSubPtcl(group+1); }


// called only on the device!
// may turn out we don't really need activePos, just get values from R directly
  KOKKOS_INLINE_FUNCTION
    void setActivePtcl(int i) {
    activePtcl(0) = i;
    activePos(0) = R(i,0);
    activePos(1) = R(i,1);
    activePos(2) = R(i,2);
    //std::cout << "doing setActivePtcl for particle " << i << std::endl;
    LikeEvaluate(i);
    //std::cout << "  finished LikeEvaluate" << std::endl;
    UnlikeEvaluate(i);
    //std::cout << "  finished UnlikeEvaluate" << std::endl;
  }

  // intended to be called by LikeDTComputeDistances and UnlikeDtComputeDistances
  // I'm just too chicken to make it private for now
  template<typename locRType, typename tempRType, typename tempDRType>
  KOKKOS_INLINE_FUNCTION
  void DTComputeDistances(RealType x0, RealType y0, RealType z0,
			  locRType& locR, tempRType& temp_r,
			  tempDRType& temp_dr,
			  int first, int last, int flip_ind = 0) {
    constexpr RealType minusone(-1);
    constexpr RealType one(1);
    for (int iat = first; iat < last; ++iat)
    {
      const RealType flip    = iat < flip_ind ? one : minusone;
      const RealType displ_0 = (locR(iat,0) - x0) * flip;
      const RealType displ_1 = (locR(iat,1) - y0) * flip;
      const RealType displ_2 = (locR(iat,2) - z0) * flip;

      const RealType ar_0 = -std::floor(displ_0 * DT_G(0,0) + displ_1 * DT_G(1,0) + displ_2 * DT_G(2,0));
      const RealType ar_1 = -std::floor(displ_0 * DT_G(0,1) + displ_1 * DT_G(1,1) + displ_2 * DT_G(2,1));
      const RealType ar_2 = -std::floor(displ_0 * DT_G(0,2) + displ_1 * DT_G(1,2) + displ_2 * DT_G(2,2));
      
      const RealType delx = displ_0 + ar_0 * DT_R(0,0) + ar_1 * DT_R(1,0) + ar_2 * DT_R(2,0);
      const RealType dely = displ_1 + ar_0 * DT_R(0,1) + ar_1 * DT_R(1,1) + ar_2 * DT_R(2,1);
      const RealType delz = displ_2 + ar_0 * DT_R(0,2) + ar_1 * DT_R(1,2) + ar_2 * DT_R(2,2);
      
      RealType rmin = delx * delx + dely * dely + delz * delz;
      int ic = 0;

      for (int c = 1; c < 8; ++c)
      {
	const RealType x  = delx + corners(c,0);
	const RealType y  = dely + corners(c,1);
	const RealType z  = delz + corners(c,2);
	const RealType r2 = x * x + y * y + z * z;
	ic         = (r2 < rmin) ? c : ic;
	rmin       = (r2 < rmin) ? r2 : rmin;
      }
     
      temp_r(iat) = std::sqrt(rmin);
      temp_dr(iat,0) = flip * (delx + corners(ic,0));
      temp_dr(iat,1) = flip * (dely + corners(ic,1));
      temp_dr(iat,2) = flip * (delz + corners(ic,2));
    }
  }

  // intended to be called by LikeDTComputeDistances and UnlikeDtComputeDistances
  // I'm just too chicken to make it private for now
  KOKKOS_INLINE_FUNCTION
  void DTComputeDistances(RealType x0, RealType y0, RealType z0,
			  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft> locR,
			  Kokkos::View<RealType*> temp_r,
			  int first, int last, int flip_ind = 0) {
    constexpr RealType minusone(-1);
    constexpr RealType one(1);
    for (int iat = first; iat < last; ++iat)
    {
      const RealType flip    = iat < flip_ind ? one : minusone;
      const RealType displ_0 = (locR(iat,0) - x0) * flip;
      const RealType displ_1 = (locR(iat,1) - y0) * flip;
      const RealType displ_2 = (locR(iat,2) - z0) * flip;

      const RealType ar_0 = -std::floor(displ_0 * DT_G(0,0) + displ_1 * DT_G(1,0) + displ_2 * DT_G(2,0));
      const RealType ar_1 = -std::floor(displ_0 * DT_G(0,1) + displ_1 * DT_G(1,1) + displ_2 * DT_G(2,1));
      const RealType ar_2 = -std::floor(displ_0 * DT_G(0,2) + displ_1 * DT_G(1,2) + displ_2 * DT_G(2,2));
      
      const RealType delx = displ_0 + ar_0 * DT_R(0,0) + ar_1 * DT_R(1,0) + ar_2 * DT_R(2,0);
      const RealType dely = displ_1 + ar_0 * DT_R(0,1) + ar_1 * DT_R(1,1) + ar_2 * DT_R(2,1);
      const RealType delz = displ_2 + ar_0 * DT_R(0,2) + ar_1 * DT_R(1,2) + ar_2 * DT_R(2,2);
      
      RealType rmin = delx * delx + dely * dely + delz * delz;

      for (int c = 1; c < 8; ++c)
      {
	const RealType x  = delx + corners(c,0);
	const RealType y  = dely + corners(c,1);
	const RealType z  = delz + corners(c,2);
	const RealType r2 = x * x + y * y + z * z;
	rmin       = (r2 < rmin) ? r2 : rmin;
      }
      temp_r(iat) = std::sqrt(rmin);
    }
  }

  
  KOKKOS_INLINE_FUNCTION
  void LikeDTComputeDistances(RealType x0, RealType y0, RealType z0, int first, int last, int flip_ind = 0) {
    DTComputeDistances(x0, y0, z0, RSoA, LikeDTTemp_r, LikeDTTemp_dr, first, last, flip_ind);
  }

  KOKKOS_INLINE_FUNCTION
  void UnlikeDTComputeDistances(RealType x0, RealType y0, RealType z0, int first, int last, int flip_ind = 0) {
    DTComputeDistances(x0, y0, z0, originR, UnlikeDTTemp_r, UnlikeDTTemp_dr, first, last, flip_ind);
  }

  KOKKOS_INLINE_FUNCTION
  RealType apply_bc(RealType x, RealType y, RealType z) {
    return x*x+y*y+z*z;
  }

  KOKKOS_INLINE_FUNCTION
  void apply_bc(Kokkos::View<RealType*[dim]> dr, Kokkos::View<RealType*> r, Kokkos::View<RealType*> rinv) const
  {
    const int n = dr.extent(0);
    constexpr RealType one(1);
    for (int i = 0; i < n; ++i)
    {
      r(i) = std::sqrt(apply_bc(dr(i,0), dr(i,1), dr(i,2)));
      rinv(i) = one / r(i);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void toUnit(RealType inX, RealType inY, RealType inZ, RealType& outX, RealType& outY, RealType& outZ) {
    outX = inX * DT_G(0,0) + inY * DT_G(1,0) + inZ * DT_G(2,0);
    outY = inX * DT_G(0,1) + inY * DT_G(1,1) + inZ * DT_G(2,1);
    outZ = inX * DT_G(0,2) + inY * DT_G(1,2) + inZ * DT_G(2,2);
  }

  KOKKOS_INLINE_FUNCTION
  void toUnit_floor(RealType inX, RealType inY, RealType inZ, RealType& outX, RealType& outY, RealType& outZ) {
    outX = inX * DT_G(0,0) + inY * DT_G(1,0) + inZ * DT_G(2,0);
    outY = inX * DT_G(0,1) + inY * DT_G(1,1) + inZ * DT_G(2,1);
    outZ = inX * DT_G(0,2) + inY * DT_G(1,2) + inZ * DT_G(2,2);
    if (-std::numeric_limits<RealType>::epsilon() < outX && outX < 0)
        outX = RealType(0.0);
      else
        outX -= std::floor(outX);
    if (-std::numeric_limits<RealType>::epsilon() < outY && outY < 0)
        outY = RealType(0.0);
      else
        outY -= std::floor(outY);
    if (-std::numeric_limits<RealType>::epsilon() < outZ && outZ < 0)
        outZ = RealType(0.0);
      else
        outZ -= std::floor(outZ);
  }

  KOKKOS_INLINE_FUNCTION
  bool outOfBound(RealType inX, RealType inY, RealType inZ) {
    if (std::abs(inX) > 0.5 || std::abs(inY) > 0.5 || std::abs(inZ) > 0.5) {
      return true;
    }
    return false;
  }

  KOKKOS_INLINE_FUNCTION
  bool isValid(RealType inX, RealType inY, RealType inZ) {
    return (BoxBConds(0) || (inX > 0.0 && inX < 1.0)) &&
      (BoxBConds(1) || (inY > 0.0 && inY < 1.0)) &&
      (BoxBConds(2) || (inZ > 0.0 && inZ < 1.0));
  }
  
  KOKKOS_INLINE_FUNCTION
  void LikeEvaluate() {
    constexpr RealType BigR = std::numeric_limits<RealType>::max();
    for (int iat = 0; iat < LikeDTDistances.extent(1); ++iat)
    {
      auto distancesSubview = Kokkos::subview(LikeDTDistances,Kokkos::ALL(),iat);
      auto displacementsSubview = Kokkos::subview(LikeDTDisplacements,Kokkos::ALL(),iat,Kokkos::ALL());
      DTComputeDistances(R(iat,0), R(iat,1), R(iat,2), RSoA,
			 distancesSubview, displacementsSubview,
			 0, LikeDTDistances.extent(1), iat);
      LikeDTDistances(iat,iat) = BigR;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void LikeEvaluate(int jat) {
    constexpr RealType BigR = std::numeric_limits<RealType>::max();
    auto distancesSubview = Kokkos::subview(LikeDTDistances,Kokkos::ALL(),jat);
    auto displacementsSubview = Kokkos::subview(LikeDTDisplacements,Kokkos::ALL(),jat,Kokkos::ALL());
    DTComputeDistances(R(jat,0), R(jat,1), R(jat,2), RSoA,
		       distancesSubview, displacementsSubview,
		       0, LikeDTDistances.extent(1), jat);
    LikeDTDistances(jat,jat) = BigR;
  }

  KOKKOS_INLINE_FUNCTION
  void UnlikeEvaluate() {
    for (int iat = 0; iat < UnlikeDTDistances.extent(1); ++iat) {
      auto distancesSubview = Kokkos::subview(UnlikeDTDistances,Kokkos::ALL(),iat);
      auto displacementsSubview = Kokkos::subview(UnlikeDTDisplacements,Kokkos::ALL(),iat,Kokkos::ALL());
      DTComputeDistances(R(iat,0), R(iat,1), R(iat,2), originR,
			 distancesSubview, displacementsSubview,
			 0, UnlikeDTDistances.extent(0));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void UnlikeEvaluate(int jat) {
    auto distancesSubview = Kokkos::subview(UnlikeDTDistances,Kokkos::ALL(),jat);
    auto displacementsSubview = Kokkos::subview(UnlikeDTDisplacements,Kokkos::ALL(),jat,Kokkos::ALL());
    //std::cout << "in UnlikeEvaluate, about to call DTComputeDistances" << std::endl;
    DTComputeDistances(R(jat,0), R(jat,1), R(jat,2), originR,
		       distancesSubview, displacementsSubview,
		       0, UnlikeDTDistances.extent(1));
  }

  KOKKOS_INLINE_FUNCTION
  void LikeMove(RealType x0, RealType y0, RealType z0) {
    LikeMoveOnSphere(x0, y0, z0);
  }

  KOKKOS_INLINE_FUNCTION
  void LikeMoveOnSphere(RealType x0, RealType y0, RealType z0) {
    LikeDTComputeDistances(x0, y0, z0, 0, LikeDTDistances.extent(0), activePtcl(0));
  }

  KOKKOS_INLINE_FUNCTION
  void UnlikeMove(RealType x0, RealType y0, RealType z0) {
    UnlikeMoveOnSphere(x0, y0, z0);
  }

  KOKKOS_INLINE_FUNCTION
  void UnlikeMoveOnSphere(RealType x0, RealType y0, RealType z0) {
    UnlikeDTComputeDistances(x0, y0, z0, 0, UnlikeDTDistances.extent(1));
  }

  KOKKOS_INLINE_FUNCTION
  void LikeUpdate(int iat) {
    for (int i = 0; i < LikeDTTemp_r.extent(0); i++) {
      LikeDTDistances(iat,i) = LikeDTTemp_r(i);
      for (int j = 0; j < dim; j++) {
	LikeDTDisplacements(iat,i,j) = LikeDTTemp_dr(i,j);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void UnlikeUpdate(int iat) {
    for (int i = 0; i < UnlikeDTTemp_r.extent(0); i++) {
      UnlikeDTDistances(iat,i) = UnlikeDTTemp_r(i);
      for (int j = 0; j < dim; j++) {
	UnlikeDTDisplacements(iat,i,j) = UnlikeDTTemp_dr(i,j);
      }
    }
  }
    
};

  
};

#endif
