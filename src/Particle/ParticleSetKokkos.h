#ifndef PARTICLE_SET_KOKKOS_H
#define PARTICLE_SET_KOKKOS_H
#include <Kokkos_Core.hpp>

namespace qmcplusplus
{

// note that some functionality provided by Lattice will be folded into this
// toUnit, outOfBound, isValid, WignerSeitzRadius?,
// toUnit_floor (whatever is needed to enable DTD_BConds)


  
// called only on the device!
// may turn out we don't really need activePos, just get values from R directly
KOKKOS_INLINE_FUNCTION
void setActivePtcl(int i) {
  activePtcl(0) = i;
  activePos(0) = R(i,0);
  activePos(1) = R(i,1);
  activePos(2) = R(i,2);
};
  
// for the moment, don't think I really need:  PCID, mass, Z, LRBox
//                                            UseSphereUpdate, SameMass, newRedPos
template<typename RealType, typename ValueType>
class ParticleSetKokkos
{
public:
  Kokkos::View<int*>        ID; // nparticles long, unique int for each particle
  Kokkos::View<int*>        IndirectID; // nparticles long (if IsGrouped=true, ID = IndirectID)   
  Kokkos::View<int*>        GroupID; // nparticles long, label to say which species
  Kokkos::View<RealType**>  R; // nparticles by dims to hold positions
  Kokkos::View<RealType**>  RSoA; // dims by nparticles (may need to explicitly specify layout)
  Kokkos::View<ValueType**> G; // nparticles by dims to hold (possibly complex) gradients
  Kokkos::View<ValueType*>  L; // nparticles long to hold (possibly complex) laplacians
  Kokkos::View<Bool*>       UseBoundBox; // single boolean value
  Kokkos::View<Bool*>       IsGrouped; // single boolean value
  Kokkos::View<int*>        activePtcl; // index pointing to the active particle for single particle moves
  Kokkos::View<RealType*>   activePos; // position of the active particle
  // See if SpeciesSet becomes necessary

  // Figure out what is necessary to carry out operations on DistTables that are called for
  // -- move, evaluate, moveOnSphere
  //   evaluate calls DTD_BConds::computeDistances,  uses R, RSoA, Distances, Displacements
  //   moveOnSphere also calls DTD_BConds::computeDistances, uses internal Temp_R and Temp_dr
  //   move uses moveOnSphere to do its work  (BA version uses Origin a lot)

  // default constructor
  KOKKOS_INLINE_FUNCTION
  ParticleSetKokkos() { ; }

  KOKKOS_INLINE_FUNCTION
  ParticleSetKokkos* operator=(const ParticleSetKokkos& rhs) {
    IS = rhs.ID;
    IndirectID = rhs.IndirectID;
    GroupID = rhs.GroupID;
    R = rhs.R;
    RSoA = rhs.RSoA;
    G = rhs.G;
    L = rhs.L;
    UseBoundBox = rhs.UseBoundBox;
    IsGrouped = rhs.IsGrouped;
    activePtcl = rhs.activePtcl;
    activePos = rhs.activePos;
  }

  ParticleSetKokkos(const ParticleSetKokkos&) = default;
  
};


}
#endif
