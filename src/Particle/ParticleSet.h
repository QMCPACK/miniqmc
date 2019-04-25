////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// D. Das, University of Illinois at Urbana-Champaign
// Bryan Clark, bclark@Princeton.edu,
//    Princeton University
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jaron T. Krogel, krogeljt@ornl.gov,
//    Oak Ridge National Laboratory
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

/** @file ParticleSet.h
 *  @brief Particle positions and related data
 */

#ifndef QMCPLUSPLUS_PARTICLESET_H
#define QMCPLUSPLUS_PARTICLESET_H

#include <Utilities/Configuration.h>
#include <Particle/Walker.h>
#include <Utilities/SpeciesSet.h>
#include <Utilities/PooledData.h>
#include <Utilities/NewTimer.h>
#include <Numerics/Containers.h>
#include <Particle/ParticleSetKokkos.h>

namespace qmcplusplus
{
/// forward declaration of DistanceTableData
class DistanceTableData;

/** Monte Carlo Data of an ensemble
 *
 * The quantities are shared by all the nodes in a group
 * - NumSamples number of samples
 * - Weight     total weight of a sample
 * - Energy     average energy of a sample
 * - Variance   variance
 * - LivingFraction fraction of walkers alive each step.
 */
template<typename T>
struct MCDataType
{
  T NumSamples;
  T RNSamples;
  T Weight;
  T Energy;
  T AlternateEnergy;
  T Variance;
  T R2Accepted;
  T R2Proposed;
  T LivingFraction;
};

/** Specialized paritlce class for atomistic simulations
 *
 * Derived from QMCTraits, ParticleBase<PtclOnLatticeTraits> and
 * OhmmsElementBase.
 * The ParticleLayout class represents a supercell with/without periodic
 * boundary
 * conditions. The ParticleLayout class also takes care of spatial
 * decompositions
 * for efficient evaluations for the interactions with a finite cutoff.
 */
class ParticleSet : public QMCTraits, public PtclOnLatticeTraits
{
public:
  using pskType = ParticleSetKokkos<RealType, ValueType, DIM>;
  /// Adding in my Kokkos ParticleSet View
  pskType psk;
  /// Function which will push data in this class to psk
  void pushDataToParticleSetKokkos();
  
  /// walker type
  typedef Walker<QMCTraits, PtclOnLatticeTraits> Walker_t;
  /// buffer type for a serialized buffer
  typedef Walker_t::Buffer_t Buffer_t;

  /// the name of the particle set.
  std::string myName;
  /// ParticleLayout
  ParticleLayout_t Lattice, PrimitiveLattice;
  /// unique, persistent ID for each particle
  ParticleIndex_t ID;
  /// index to the primitice cell with tiling
  ParticleIndex_t PCID;
  /// Species ID
  ParticleIndex_t GroupID;
  /// Position
  ParticlePos_t R;
  /// SoA copy of R
  VectorSoAContainer<RealType, DIM> RSoA;
  /// gradients of the particles
  ParticleGradient_t G;
  /// laplacians of the particles
  ParticleLaplacian_t L;
  /** ID map that reflects species group
   *
   * IsGrouped=true, if ID==IndirectID
   */
  ParticleIndex_t IndirectID;
  /// mass of each particle
  ParticleScalar_t Mass;
  /// charge of each particle
  ParticleScalar_t Z;

  /// Long-range box
  ParticleLayout_t LRBox;
  /// true, if a physical or local bounding box is used
  bool UseBoundBox;
  /// true if fast update for sphere moves
  bool UseSphereUpdate;
  /// true if the particles are grouped
  bool IsGrouped;
  /// true if the particles have the same mass
  bool SameMass;
  /// the index of the active particle for particle-by-particle moves
  Index_t activePtcl;

  /** the position of the active particle for particle-by-particle moves
   *
   * Saves the position before making a move to handle rejectMove
   */
  SingleParticlePos_t activePos;

  /** the proposed position in the Lattice unit
   */
  SingleParticlePos_t newRedPos;

  /// SpeciesSet of particles
  SpeciesSet mySpecies;

  /// distance tables that need to be updated by moving this ParticleSet
  std::vector<DistanceTableData*> DistTables;

  /// current MC step
  int current_step;


  /// default constructor
  ParticleSet();

  /// copy constructor
  ParticleSet(const ParticleSet& p);

  /// default destructor
  virtual ~ParticleSet();

  /** create  particles
   * @param numPtcl number of particles
   */
  void create(int numPtcl);
  /** create grouped particles
   * @param agroup number of particles per group
   */
  void create(const std::vector<int>& agroup);

  /// write to a std::ostream
  bool get(std::ostream&) const;

  /// read from std::istream
  bool put(std::istream&);

  /// reset member data
  void reset();

  /// set UseBoundBox
  void setBoundBox(bool yes);

  /**  add a distance table
   * @param psrc source particle set
   *
   * Ensure that the distance for this-this is always created first.
   */
  int addTable(const ParticleSet& psrc, int dt_type);

  /** update the internal data
   *@param skip SK update if skipSK is true
   */
  void update(bool skipSK = false);

  /// retrun the SpeciesSet of this particle set
  inline SpeciesSet& getSpeciesSet() { return mySpecies; }
  /// retrun the const SpeciesSet of this particle set
  inline const SpeciesSet& getSpeciesSet() const { return mySpecies; }

  inline void setName(const std::string& aname) { myName = aname; }

  /// return the name
  inline const std::string& getName() const { return myName; }

  void resetGroups();

  /** set active particle
   * @param iat particle index
   *
   * Compute internal data based on current R[iat]
   * Introduced to work with update-only methods.
   */
  void setActive(int iat);

  void multi_setActiveKokkos(std::vector<ParticleSet*>& P_list, int iel);
  /** return the position of the active partice
   *
   * activePtcl=-1 is used to flag non-physical moves
   */
  inline const PosType& activeR(int iat) const { return (activePtcl == iat) ? activePos : R[iat]; }

  /** move a particle
   * @param iat the index of the particle to be moved
   * @param displ random displacement of the iat-th particle
   * @return true, if the move is valid
   */
  bool makeMoveAndCheck(Index_t iat, const SingleParticlePos_t& displ);


  void multi_makeMoveAndCheckKokkos(std::vector<ParticleSet*>& P_list, Kokkos::View<RealType*[3]>& dr,
				    int iel, std::vector<int> isValid);
  /** move a particle
   * @param iat the index of the particle to be moved
   * @param displ random displacement of the iat-th particle
   */
  void makeMoveOnSphere(Index_t iat, const SingleParticlePos_t& displ);

  /** accept the move
   *@param iat the index of the particle whose position and other attributes to
   *be updated
   */
  void acceptMove(Index_t iat);

  /** reject the move
   */
  void rejectMove(Index_t iat);

  void multi_acceptRejectMoveKokkos(std::vector<ParticleSet*>& psets, 
				    std::vector<bool>& isAccepted, int iel);

  void clearDistanceTables();

  void convert2Unit(ParticlePos_t& pout);
  void convert2Cart(ParticlePos_t& pout);

  /** load a Walker_t to the current ParticleSet
   * @param awalker the reference to the walker to be loaded
   * @param pbyp true if it is used by PbyP update
   *
   * PbyP requires the distance tables and Sk with awalker.R
   */
  void loadWalker(Walker_t& awalker, bool pbyp);
  /** save this to awalker
   */
  void saveWalker(Walker_t& awalker);

  /** update the buffer
   *@param skip SK update if skipSK is true
   */
  void donePbyP(bool skipSK = false);

  void multi_donePbyP(std::vector<ParticleSet*>& psets, bool skipSK = false);


  template<typename allPsdType, typename EiListType, typename rOnSphereType,
           typename bigElPosType, typename tempRType>    
  void updateTempPosAndRs(int eiPair, allPsdType& allParticleSetData,
			  EiListType& EiLists, rOnSphereType& rOnSphere, bigElPosType& bigElPos,
			  tempRType& bigLikeTempR, tempRType& bigUnlikeTempR) {
    int eiPair_ = eiPair;
    auto allParticleSetData_ = allParticleSetData;
    auto EiLists_ = EiLists;
    auto rOnSphere_ = rOnSphere;
    auto bigElPos_ = bigElPos;
    auto bigLikeTempR_ = bigLikeTempR;
    auto bigUnlikeTempR_ = bigUnlikeTempR;
    
    const int numMovers = allParticleSetData_.extent(0);
    const int numKnots = rOnSphere_.extent(0);
    Kokkos::TeamPolicy<> pol(numMovers, 1, 32);
    Kokkos::parallel_for("updateTempPosAndRs", pol,
			 KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
			   const int walkerNum = member.league_rank();
			   const int eNum = EiLists_(walkerNum, eiPair, 0);
			   const int atNum = EiLists_(walkerNum, eiPair, 1);
			   if (eNum > -1) {
			     Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, numKnots),
						  [=](const int& knotNum) {
						    // handles bigElPos
						    for (int dim = 0; dim < 3; dim++) {
						      bigElPos_(walkerNum, knotNum, dim) = allParticleSetData_(walkerNum).UnlikeDTDistances(eNum,atNum) *
							rOnSphere_(walkerNum,knotNum,dim) - allParticleSetData_(walkerNum).UnlikeDTDisplacements(eNum,atNum,dim);
						    }
						    // do bigLikeTempR
						    auto likeTempRSubview = Kokkos::subview(bigLikeTempR_,walkerNum,knotNum,Kokkos::ALL());
						    auto unlikeTempRSubview = Kokkos::subview(bigUnlikeTempR_,walkerNum,knotNum,Kokkos::ALL());
						    allParticleSetData_(walkerNum).DTComputeDistances(bigElPos(walkerNum,knotNum,0),	   
												      bigElPos(walkerNum,knotNum,1),	   
												      bigElPos(walkerNum,knotNum,2),	   
												      allParticleSetData_(walkerNum).RSoA,			   
												      likeTempRSubview,		   
												      0, likeTempRSubview.extent(0), eNum);
						    allParticleSetData_(walkerNum).DTComputeDistances(bigElPos(walkerNum,knotNum,0),	   
												      bigElPos(walkerNum,knotNum,1),	   
												      bigElPos(walkerNum,knotNum,2),	   
												      allParticleSetData_(walkerNum).originR,			   
												      unlikeTempRSubview,		   
												      0, unlikeTempRSubview.extent(0));
						  });
			   }
			 });
  }
						    
  inline void setTwist(SingleParticlePos_t& t) { myTwist = t; }
  inline SingleParticlePos_t getTwist() const { return myTwist; }

  /** get species name of particle i
   */
  inline const std::string& species_from_index(int i) { return mySpecies.speciesName[GroupID[i]]; }

  inline int getTotalNum() const { return TotalNum; }

  inline void resize(int numPtcl)
  {
    TotalNum = numPtcl;

    R.resize(numPtcl);
    ID.resize(numPtcl);
    PCID.resize(numPtcl);
    GroupID.resize(numPtcl);
    G.resize(numPtcl);
    L.resize(numPtcl);
    Mass.resize(numPtcl);
    Z.resize(numPtcl);
    IndirectID.resize(numPtcl);

    RSoA.resize(numPtcl);
  }

  inline void assign(const ParticleSet& ptclin)
  {
    resize(ptclin.getTotalNum());
    Lattice          = ptclin.Lattice;
    PrimitiveLattice = ptclin.PrimitiveLattice;
    R.InUnit         = ptclin.R.InUnit;
    R                = ptclin.R;
    ID               = ptclin.ID;
    GroupID          = ptclin.GroupID;
    if (ptclin.SubPtcl.size())
    {
      SubPtcl.resize(ptclin.SubPtcl.size());
      SubPtcl = ptclin.SubPtcl;
    }
  }

  /// return the number of groups
  inline int groups() const { return SubPtcl.size() - 1; }

  /// return the first index of a group i
  inline int first(int igroup) const { return SubPtcl[igroup]; }

  /// return the last index of a group i
  inline int last(int igroup) const { return SubPtcl[igroup + 1]; }
  
protected:
  /** map to handle distance tables
   *
   * myDistTableMap[source-particle-tag]= locator in the distance table
   * myDistTableMap[ObjectTag] === 0
   */
  std::map<std::string, int> myDistTableMap;

  SingleParticlePos_t myTwist;

  /// total number of particles
  int TotalNum;

  /// array to handle a group of distinct particles per species
  ParticleIndex_t SubPtcl;

  /// Timer
  TimerList_t timers;
};

const std::vector<ParticleSet::ParticleGradient_t*>
    extract_G_list(const std::vector<ParticleSet*>& P_list);
const std::vector<ParticleSet::ParticleLaplacian_t*>
    extract_L_list(const std::vector<ParticleSet*>& P_list);

} // namespace qmcplusplus
#endif
