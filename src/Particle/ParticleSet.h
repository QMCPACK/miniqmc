//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: D. Das, University of Illinois at Urbana-Champaign
//                    Bryan Clark, bclark@Princeton.edu, Princeton University
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    



#ifndef QMCPLUSPLUS_PARTICLESET_H
#define QMCPLUSPLUS_PARTICLESET_H

#include <Configuration.h>
#include <Particle/Walker.h>
#include <Utilities/SpeciesSet.h>
#include <Utilities/PooledData.h>
#include <OhmmsPETE/OhmmsArray.h>
#include <OhmmsSoA/Container.h>

namespace qmcplusplus
{

///forward declaration of DistanceTableData
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
 * Derived from QMCTraits, ParticleBase<PtclOnLatticeTraits> and OhmmsElementBase.
 * The ParticleLayout class represents a supercell with/without periodic boundary
 * conditions. The ParticleLayout class also takes care of spatial decompositions
 * for efficient evaluations for the interactions with a finite cutoff.
 */
class ParticleSet
  :  public QMCTraits
  , public PtclOnLatticeTraits
{
public:
  ///@typedef walker type
  typedef Walker<QMCTraits,PtclOnLatticeTraits> Walker_t;
  ///@typedef buffer type for a serialized buffer
  typedef Walker_t::Buffer_t             Buffer_t;

  ///the name of the particle set.
  std::string myName;
  //!< ParticleLayout
  ParticleLayout_t Lattice, PrimitiveLattice;
  //!< unique, persistent ID for each particle
  ParticleIndex_t ID;
  ///index to the primitice cell with tiling
  ParticleIndex_t PCID;
  //!< Species ID
  ParticleIndex_t GroupID;
  //!< Position
  ParticlePos_t R;
  ///SoA copy of R
  VectorSoaContainer<RealType,DIM> RSoA;
  ///gradients of the particles
  ParticleGradient_t G;
  ///laplacians of the particles
  ParticleLaplacian_t L;
  ///differential gradients of the particles
  ParticleGradient_t dG;
  ///differential laplacians of the particles
  ParticleLaplacian_t dL;
  /** ID map that reflects species group
   *
   * IsGrouped=true, if ID==IndirectID
   */
  ParticleIndex_t IndirectID;
  ///mass of each particle
  ParticleScalar_t Mass;
  ///charge of each particle
  ParticleScalar_t Z;

  ///Long-range box
  ParticleLayout_t LRBox;
  ///true, if a physical or local bounding box is used
  bool UseBoundBox;
  ///true if fast update for sphere moves
  bool UseSphereUpdate;
  ///true if the particles are grouped
  bool IsGrouped;
  ///true if the particles have the same mass
  bool SameMass;
  /// true if all the internal state is ready for estimators
  bool Ready4Measure;
  ///threa id
  Index_t ThreadID;
  ///the index of the active particle for particle-by-particle moves
  Index_t activePtcl;
  ///the group of the active particle for particle-by-particle moves
  Index_t activeGroup;
  ///the index of the active bead for particle-by-particle moves
  Index_t activeBead;
  ///the direction reptile traveling
  Index_t direction;
  ///pointer to the working walker
  Walker_t*  activeWalker;

  /** the position of the active particle for particle-by-particle moves
   *
   * Saves the position before making a move to handle rejectMove
   */
  SingleParticlePos_t activePos;

  /** the proposed position in the Lattice unit
   */
  SingleParticlePos_t newRedPos;

  ///SpeciesSet of particles
  SpeciesSet mySpecies;

  ///distance tables that need to be updated by moving this ParticleSet
  std::vector<DistanceTableData*> DistTables;

  ///spherical-grids for non-local PP
  std::vector<ParticlePos_t*> Sphere;

  ///Particle density in G-space for MPC interaction
  std::vector<TinyVector<int,OHMMS_DIM> > DensityReducedGvecs;
  std::vector<ComplexType>   Density_G;
  Array<RealType,OHMMS_DIM> Density_r;

  /// DFT potential
  std::vector<TinyVector<int,OHMMS_DIM> > VHXCReducedGvecs;
  std::vector<ComplexType>   VHXC_G[2];
  Array<RealType,OHMMS_DIM> VHXC_r[2];

  ///clones of this object: used by the thread pool
  std::vector<ParticleSet*> myClones;

  ///current MC step
  int current_step;

  ///default constructor
  ParticleSet();

  ///copy constructor
  ParticleSet(const ParticleSet& p);

  ///default destructor
  virtual ~ParticleSet();

  /** create  particles
   * @param numPtcl number of particles
   */
  void create(int numPtcl);
  /** create grouped particles
   * @param agroup number of particles per group
   */
  void create(const std::vector<int>& agroup);

  ///write to a std::ostream
  bool get(std::ostream& ) const;

  ///read from std::istream
  bool put( std::istream& );

  ///reset member data
  void reset();

  ///set UseBoundBox
  void setBoundBox(bool yes);

  /** check bounding box
   * @param rb cutoff radius to check the condition
   */
  void checkBoundBox(RealType rb);

  /**  add a distance table
   * @param psrc source particle set
   *
   * Ensure that the distance for this-this is always created first.
   */
  int  addTable(const ParticleSet& psrc, int dt_type);

  /** returns index of a distance table, -1 if not present
   * @param psrc source particle set
   */
  int getTable(const ParticleSet& psrc);

  /** update the internal data
   *@param skip SK update if skipSK is true
   */
  void update(bool skipSK=false);

  /**update the internal data with new position
   *@param pos position vector assigned to R
   */
  void update(const ParticlePos_t& pos);

  ///retrun the SpeciesSet of this particle set
  inline SpeciesSet& getSpeciesSet()
  {
    return mySpecies;
  }
  ///retrun the const SpeciesSet of this particle set
  inline const SpeciesSet& getSpeciesSet() const
  {
    return mySpecies;
  }

  ///return this id
  inline int tag() const
  {
    return ObjectTag;
  }

  ///return parent's id
  inline int parent() const
  {
    return ParentTag;
  }

  ///return parent's name
  inline const std::string& parentName() const
  {
    return ParentName;
  }

  inline void setName(const std::string& aname)
  {
    myName     = aname;
    if(ParentName=="0")
    {
      ParentName = aname;
    }
  }

  ///return the name
  inline const std::string& getName() const
  {
    return myName;
  }

  void resetGroups();

  /** set active particle
   * @param iat particle index
   *
   * Compute internal data based on current R[iat]
   * Introduced to work with update-only methods.
   */
  void setActive(int iat);
  
  /**move a particle
   *@param iat the index of the particle to be moved
   *@param displ random displacement of the iat-th particle
   *
   * Update activePos  by  R[iat]+displ
   */
  SingleParticlePos_t makeMove(Index_t iat, const SingleParticlePos_t& displ);

  /** move a particle
   * @param iat the index of the particle to be moved
   * @param displ random displacement of the iat-th particle
   * @return true, if the move is valid
   */
  bool makeMoveAndCheck(Index_t iat, const SingleParticlePos_t& displ);

  /** move all the particles of a walker
   * @param awalker the walker to operate
   * @param deltaR proposed displacement
   * @param dt  factor of deltaR
   * @return true if all the moves are legal.
   *
   * If big displacements or illegal positions are detected, return false.
   * If all good, R = awalker.R + dt* deltaR
   */
  bool makeMove(const Walker_t& awalker, const ParticlePos_t& deltaR, RealType dt);

  bool makeMove(const Walker_t& awalker, const ParticlePos_t& deltaR, const std::vector<RealType>& dt);
  /** move all the particles including the drift
   *
   * Otherwise, everything is the same as makeMove for a walker
   */
  bool makeMoveWithDrift(const Walker_t& awalker
                         , const ParticlePos_t& drift, const ParticlePos_t& deltaR, RealType dt);

  bool makeMoveWithDrift(const Walker_t& awalker
                         , const ParticlePos_t& drift, const ParticlePos_t& deltaR, const std::vector<RealType>& dt);

  void makeMoveOnSphere(Index_t iat, const SingleParticlePos_t& displ);

  /** Handles a virtual move for all the particles to ru.
   * @param ru position in the reduced cordinate
   *
   * The data of the 0-th particle is overwritten by the new position
   * and the rejectMove should be called for correct use.
   * See QMCHamiltonians::MomentumEstimator
   */
  void makeVirtualMoves(const SingleParticlePos_t& newpos);

  /** accept the move
   *@param iat the index of the particle whose position and other attributes to be updated
   */
  void acceptMove(Index_t iat);

  /** reject the move
   */
  void rejectMove(Index_t iat);

  inline SingleParticlePos_t getOldPos() const
  {
    return activePos;
  }

  void clearDistanceTables();
  void resizeSphere(int nc);

  void convert(const ParticlePos_t& pin, ParticlePos_t& pout);
  void convert2Unit(const ParticlePos_t& pin, ParticlePos_t& pout);
  void convert2Cart(const ParticlePos_t& pin, ParticlePos_t& pout);
  void convert2Unit(ParticlePos_t& pout);
  void convert2Cart(ParticlePos_t& pout);
  void convert2UnitInBox(const ParticlePos_t& pint, ParticlePos_t& pout);
  void convert2CartInBox(const ParticlePos_t& pint, ParticlePos_t& pout);

  void applyBC(const ParticlePos_t& pin, ParticlePos_t& pout);
  void applyBC(ParticlePos_t& pos);
  void applyBC(const ParticlePos_t& pin, ParticlePos_t& pout, int first, int last);
  void applyMinimumImage(ParticlePos_t& pinout);

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

  /** load a walker : R <= awalker->R 
   *
   * No other copy is made
   */
  void loadWalker(Walker_t* awalker);

  /** update the buffer
   *@param skip SK update if skipSK is true
   */
  void donePbyP(bool skipSK=false);

  inline void setTwist(SingleParticlePos_t& t)
  {
    myTwist=t;
  }
  inline SingleParticlePos_t getTwist() const
  {
    return myTwist;
  }

  /** return the ip-th clone
   * @param ip thread number
   *
   * Return itself if ip==0
   */
  inline ParticleSet* get_clone(int ip)
  {
    if(ip >= myClones.size())
      return 0;
    return (ip)? myClones[ip]:this;
  }

  inline const ParticleSet* get_clone(int ip) const
  {
    if(ip >= myClones.size())
      return 0;
    return (ip)? myClones[ip]:this;
  }

  inline int clones_size() const
  {
    return myClones.size();
  }

  /** update R of its own and its clones
   * @param rnew new position array of N
   */
  template<typename PAT>
  inline void update_clones(const PAT& rnew)
  {
    if(R.size() != rnew.size())
      APP_ABORT("ParticleSet::updateR failed due to different sizes");
    R=rnew;
    for(int ip=1; ip<myClones.size(); ++ip)
      myClones[ip]->R=rnew;
  }

  /** reset internal data of clones including itself
   */
  void reset_clones();
      
  /** get species name of particle i
   */
  inline const std::string& species_from_index(int i)
  {
    return mySpecies.speciesName[GroupID[i]];
  }

  inline int getTotalNum() const
  {
    return TotalNum;
  }

  inline void resize(int numPtcl)
  {
    TotalNum = numPtcl;

    R.resize(numPtcl);
    ID.resize(numPtcl);
    PCID.resize(numPtcl);
    GroupID.resize(numPtcl);
    G.resize(numPtcl);
    dG.resize(numPtcl);
    L.resize(numPtcl);
    dL.resize(numPtcl);
    Mass.resize(numPtcl);
    Z.resize(numPtcl);
    IndirectID.resize(numPtcl);

    RSoA.resize(numPtcl);
  }

  inline void assign(const ParticleSet& ptclin)
  {
    TotalNum = ptclin.getTotalNum();
    resize(TotalNum);
    Lattice = ptclin.Lattice;
    PrimitiveLattice = ptclin.PrimitiveLattice;
    R.InUnit = ptclin.R.InUnit;
    R = ptclin.R;
    ID = ptclin.ID;
    GroupID = ptclin.GroupID;
    if(ptclin.SubPtcl.size())
    {
      SubPtcl.resize(ptclin.SubPtcl.size());
      SubPtcl =ptclin.SubPtcl;
    }
  }

  ///return the number of groups
  inline int groups() const
  {
    return SubPtcl.size()-1;
  }

  ///return the first index of a group i
  inline int first(int igroup) const
  {
    return SubPtcl[igroup];
  }

  ///return the last index of a group i
  inline int last(int igroup) const
  {
    return SubPtcl[igroup+1];
  }

protected:
  ///the number of particle objects
  static Index_t PtclObjectCounter;

  ///id of this object
  Index_t ObjectTag;

  ///id of the parent
  Index_t ParentTag;

  /** map to handle distance tables
   *
   * myDistTableMap[source-particle-tag]= locator in the distance table
   * myDistTableMap[ObjectTag] === 0
   */
  std::map<int,int> myDistTableMap;
  void initParticleSet();

  SingleParticlePos_t myTwist;

  std::string ParentName;

  ///total number of particles
  int TotalNum;

  ///array to handle a group of distinct particles per species
  ParticleIndex_t                       SubPtcl;

};
}
#endif
