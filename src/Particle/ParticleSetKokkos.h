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
  Kokkos::View<Bool[1]>                                   UseBoundBox; // single boolean value
  Kokkos::View<Bool[1]>                                   IsGrouped; // single boolean value
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
  
  Kokkos::View<RealType**>                              LikeDTDistances; // [nparticles][nparticles]
  Kokkos::View<RealType**[dim]>                         LikeDTDisplacements; // [nparticles][nparticles][dim]
  Kokkos::View<RealType*>                               LikeDTTemp_r; // [nparticles]
  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft>       LikeDTTemp_dr; // [nparticles][dim]

  Kokkos::View<RealType**>                              UnlikeDTDistances; // [nparticles][ntargets]
  Kokkos::View<RealType**[dim]>                         UnlikeDTDisplacements; // [nparticles][ntargets][dim]
  Kokkos::View<RealType*>                               UnlikeDTTemp_r; // [ntargets]
  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft>       UnlikeDTTemp_dr; // [ntargets][dim]

  Kokkos::View<RealType*[dim],Kokkos::LayoutLeft>       originR; // [ntargets][dim]  // locations of all the particles in the source set
  
  // default constructor
  KOKKOS_INLINE_FUNCTION
  ParticleSetKokkos() { ; }

  KOKKOS_INLINE_FUNCTION
  ParticleSetKokkos* operator=(const ParticleSetKokkos& rhs) {
    ID = rhs.ID;
    IndirectID = rhs.IndirectID;
    GroupID = rhs.GroupID;
    SubPtcl = rhs.SubPtcl;
    R = rhs.R;
    RSoA = rhs.RSoA;
    G = rhs.G;
    L = rhs.L;
    UseBoundBox = rhs.UseBoundBox;
    IsGrouped = rhs.IsGrouped;
    activePtcl = rhs.activePtcl;
    activePos = rhs.activePos;
    DT_G = rhs.DT_G;
    DT_R = rhs.DT_R;
    corners = rhs.corners;
    LikeDTDDistances = rhs.LikeDTDDistances;
    LikeDTDisplacements = rhs.LikeDTDisplacements;
    LikeDTTemp_r = rhs.LikeDTTemp_r;
    LikeDTTemp_dr = rhs.LikeDTTemp_dr;
    originR = rhs.originR;
  }

  ParticleSetKokkos(const ParticleSetKokkos&) = default;

  KOKKOS_INLINE_FUNCTION
    int first(int group) const { return SubPtcl(group); }

  KOKKOS_INLINE_FUNCTION
    int last(int group) const { return SubPtcl(group+1); }

// called only on the device!
// may turn out we don't really need activePos, just get values from R directly
  KOKKOS_INLINE_FUNCTION
    void setActivePtcl(int i) {
    activePtcl(0) = i;
    activePos(0) = R(i,0);
    activePos(1) = R(i,1);
    activePos(2) = R(i,2);
  }

 void computeDistances(const PT& pos,
                        const RSoA& R0,
                        T* restrict temp_r,
                        RSoA& temp_dr,
                        int first,
                        int last,
                        int flip_ind = 0)

  KOKKOS_INLINE_FUNCTION
  LikeDTDComputeDistances(real_type x0, real_type y0, real_type z0, int first, int last, int flip_ind = 0) {
   constexpr real_type minusone(-1);
   constexpr real_type one(1);
   for (int iat = first; iat < last; ++iat)
   {
     const real_type flip    = iat < flip_ind ? one : minusone;
     const real_type displ_0 = (RSoA(iat,0) - x0) * flip;
     const real_type displ_1 = (RSoA(iat,1) - y0) * flip;
     const real_type displ_2 = (RSoA(iat,2) - z0) * flip;

     const real_type ar_0 = -std::floor(displ_0 * DT_G(0,0) + displ_1 * DT_G(1,0) + displ_2 * DT_G(2,0));
     const real_type ar_1 = -std::floor(displ_0 * DT_G(0,1) + displ_1 * DT_G(1,1) + displ_2 * DT_G(2,1));
     const real_type ar_2 = -std::floor(displ_0 * DT_G(0,2) + displ_1 * DT_G(1,2) + displ_2 * DT_G(2,2));

     const real_type delx = displ_0 + ar_0 * DT_R(0,0) + ar_1 * DT_R(1,0) + ar_2 * DT_R(2,0);
     const real_type dely = displ_1 + ar_0 * DT_R(0,1) + ar_1 * DT_R(1,1) + ar_2 * DT_R(2,1);
     const real_type delz = displ_2 + ar_0 * DT_R(0,2) + ar_1 * DT_R(1,2) + ar_2 * DT_R(2,2);

     real_type rmin = delx * delx + dely * dely + delz * delz;
     int ic = 0;
     
     for (int c = 1; c < 8; ++c)
     {
       const real_type x  = delx + corners(c,0);
       const real_type y  = dely + corners(c,1);
       const real_type z  = delz + corners(c,2);
       const real_type r2 = x * x + y * y + z * z;
       ic         = (r2 < rmin) ? c : ic;
       rmin       = (r2 < rmin) ? r2 : rmin;
     }
     
     LikeDTTemp_r(iat) = std::sqrt(rmin);
     LikeDTTemp_dr(iat,0) = flip * (delx + corners(ic,0));
     LikeDTTemp_dr(iat,1) = flip * (dely + corners(ic,1));
     LikeDTTemp_dr(iat,2) = flip * (delz + corners(ic,2));
   }
 }

  KOKKOS_INLINE_FUNCTION
  UnlikeDTDComputeDistances(real_type x0, real_type y0, real_type z0, int first, int last, int flip_ind = 0) {
   constexpr real_type minusone(-1);
   constexpr real_type one(1);
   for (int iat = first; iat < last; ++iat)
   {
     const real_type flip    = iat < flip_ind ? one : minusone;
     const real_type displ_0 = (OriginR(iat,0) - x0) * flip;
     const real_type displ_1 = (OriginR(iat,1) - y0) * flip;
     const real_type displ_2 = (OriginR(iat,2) - z0) * flip;

     const real_type ar_0 = -std::floor(displ_0 * DT_G(0,0) + displ_1 * DT_G(1,0) + displ_2 * DT_G(2,0));
     const real_type ar_1 = -std::floor(displ_0 * DT_G(0,1) + displ_1 * DT_G(1,1) + displ_2 * DT_G(2,1));
     const real_type ar_2 = -std::floor(displ_0 * DT_G(0,2) + displ_1 * DT_G(1,2) + displ_2 * DT_G(2,2));

     const real_type delx = displ_0 + ar_0 * DT_R(0,0) + ar_1 * DT_R(1,0) + ar_2 * DT_R(2,0);
     const real_type dely = displ_1 + ar_0 * DT_R(0,1) + ar_1 * DT_R(1,1) + ar_2 * DT_R(2,1);
     const real_type delz = displ_2 + ar_0 * DT_R(0,2) + ar_1 * DT_R(1,2) + ar_2 * DT_R(2,2);

     real_type rmin = delx * delx + dely * dely + delz * delz;
     int ic = 0;
     
     for (int c = 1; c < 8; ++c)
     {
       const real_type x  = delx + corners(c,0);
       const real_type y  = dely + corners(c,1);
       const real_type z  = delz + corners(c,2);
       const real_type r2 = x * x + y * y + z * z;
       ic         = (r2 < rmin) ? c : ic;
       rmin       = (r2 < rmin) ? r2 : rmin;
     }
     
     UnlikeDTTemp_r(iat) = std::sqrt(rmin);
     UnlikeDTTemp_dr(iat,0) = flip * (delx + corners(ic,0));
     UnlikeDTTemp_dr(iat,1) = flip * (dely + corners(ic,1));
     UnlikeDTTemp_dr(iat,2) = flip * (delz + corners(ic,2));
   }
 }

};

  
};

#endif
