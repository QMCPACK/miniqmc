#include<Kokkos_Core.hpp>

// Define Scalar Types (incomplete list of all scalars below)
#ifndef T_INT
define T_INT int
#endif

#ifndef T_FLOAT
#define T_FLOAT double
#endif

// Define Kokkos View Types (incomplete list of all arrays below)
typedef Kokkos::View<T_INT*>      IntVectorType_t;
typedef Kokkos::View<T_FLOAT*>   FloatViewVector_t;
typedef Kokkos::View<T_INT**>     IntMatrixType_t;
typedef Kokkos::View<T_FLOAT**>  FloatMatrixType_t;

/*IntMatrixType_t tmat("tmat", D, D); // src/miniapps/einspline_spo.cpp: Tensor<int,3> tmat(na,0,0,0,nb,0,0,0,nc), D=3;
ViewMatrixType_t X("X", D, D); // src/OhmmsPete/Tensor.h: Tensor<int,3> tmat(na,0,0,0,nb,0,0,0,nc), D=3;
ViewMatrixType_double lattice_b("lattice_b", D, D); 
ViewMatrixType_double tile_graphite("tile_graphite", D, D); //src/miniapps/einspline_spo.cpp: Tensor<T,3>  tile_graphite(ParticleSet& ions, Tensor<int,3>& tmat, T scale), D=3;
ViewMatrixType_double R("R", numPtcl, D); //graphite.hpp: 29: ions.create(4), numPtcl=4, D=3;
ViewMatrixType_int I("I", D, D); // src/Particle/ParticleIOUtility.h: Tensor<int,3> I(1,0,0,0,1,0,0,0,1);
*/

// These need to be appended to Kokkos View Types
// List of all data structures called from einspline_spo.cpp
//
// Stopped at src/einspline_spo.cpp: 109: lattice_b=tile_graphite(ions,tmat,scale); 
//
// src/miniapps/einspline_spo.cpp:
//   tmat: 3 x 3 matrix.
//     Tensor<int,3> tmat(na,0,0,0,nb,0,0,0,nc);
//   lattice_b: 3 x 3 matrix.
//     Tensor<OHMMS_PRECISION,3> lattice_b;
//   tile_graphite: 3 x 3 matrix.
//     Tensor<T,3>  tile_graphite(ParticleSet& ions, Tensor<int,3>& tmat, T scale)
//
// src/Lattice/CrystalLattice.cpp:
//   R: matrix:  D x D 
//   G: matrix:  D x D 
//   Rv: matrix: D x D
//   Gv: matrix: D x D
//   Length: vector: D
//   OneOverLength: vector: D
//   Center: vector: D
//   M: matrix: D x D
//   Mg: matrix: D x D 
//   BoxBConds: vector: D
//
// src/Particle/ParticleIOUtility.h:
//   primTypes: vector: iat
//   uPrim: vector: 3
//   uSuper: vector: 3
//   primPos: matrix: iat x 3, iat=4 (suspect iat=numPtcl)
//   r: vector: 3
//
// src/graphite.hpp: 
//   R: numPtcl x 3 matrix
//     ions.create(4);
//     src/Particle/ParticleSet: ParticleSet::create(int numPtcl) 
//   I: 3 x 3 matrix
//     src/Particle/ParticleIOUtility.h: 
//     Tensor<int,3> I(1,0,0,0,1,0,0,0,1);
//
// unknown:  
//   ParticleSet ions; (einspline_spo.cpp: 107) 
//
//
// Comments:
//   Define type: OHMMS_PRECISION.
//     OHMMS_PRECISION seems to be type double.
//
//
// List of scalars 
// einspline_spo.cpp:
//   int: 
//     na;             
//     nb;             
//     nc;             
//     nsteps;       
//     iseed;         
//     nx;
//     ny;
//     nz;
//     tileSize;
//     ncrews;
//     opt;
//     nptcl;   
//     nknots_copy;
//     nTiles;
//     nions;
//     nels;
//     tileSize;
//     nTiles;
//     time_array_size;
//     np;
//     ip;
//     teamID;
//     crewID;
//
//   double:
//     t0;
//     tInit;
//     vgh_t;
//     val_t;
//     nspheremoves;
//     dNumVGHCalls;
