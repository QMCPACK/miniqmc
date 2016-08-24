
#include "Message/catch_mpi_main.hpp"

#include "OhmmsData/Libxml2Doc.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "Utilities/OhmmsInfo.h"
#include "Lattice/ParticleBConds.h"
#include "Particle/ParticleSet.h"
#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "Particle/SymmetricDistanceTableData.h"
#include "QMCApp/ParticleSetPool.h"
#include "QMCWaveFunctions/OrbitalBase.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/Jastrow/PadeJastrowBuilder.h"


#include <stdio.h>
#include <string>

using std::string;

namespace qmcplusplus
{

TEST_CASE("Pade functor", "[wavefunction]")
{

  double A = -0.25;
  double B = 0.1;
  PadeFunctor<double> pf(A, B);

  double r = 1.2;
  double u = pf.evaluate(r);
  REQUIRE(u == 2.232142857142857);
}

TEST_CASE("Pade Jastrow", "[wavefunction]")
{

    Communicate *c;
    OHMMS::Controller->initialize(0, NULL);
    c = OHMMS::Controller;
    OhmmsInfo("testlogfile");

    ParticleSet ions_;
    ParticleSet elec_;

    ions_.setName("ion");
    ions_.R.resize(1);
    ions_.R[0][0] = 0.0;
    ions_.R[0][1] = 0.0;
    ions_.R[0][2] = 0.0;
    ions_.create(1);

    elec_.setName("elec");
    elec_.R.resize(2);
    elec_.R[0][0] = -0.28;
    elec_.R[0][1] = 0.0225;
    elec_.R[0][2] = -2.709;
    elec_.R[1][0] = -1.08389;
    elec_.R[1][1] = 1.9679;
    elec_.R[1][2] = -0.0128914;
    elec_.create(2);

    SpeciesSet &tspecies =  elec_.getSpeciesSet();
    int upIdx = tspecies.addSpecies("u");
    int downIdx = tspecies.addSpecies("d");
    int chargeIdx = tspecies.addAttribute("charge");
    tspecies(chargeIdx, upIdx) = -1;
    tspecies(chargeIdx, downIdx) = -1;

    elec_.addTable(ions_);
    elec_.update();


  TrialWaveFunction psi = TrialWaveFunction(c);
  // Need 1 electron and 1 proton, somehow
  //ParticleSet target = ParticleSet();
  ParticleSetPool ptcl = ParticleSetPool(c);

const char *particles = \
"<tmp> \
<jastrow name=\"Jee\" type=\"Two-Body\" function=\"pade\"> \
  <correlation speciesA=\"u\" speciesB=\"u\"> \
        <var id=\"juu_b\" name=\"B\">0.1</var> \
  </correlation> \
</jastrow> \
</tmp> \
";
  Libxml2Document doc;
  bool okay = doc.parseFromString(particles);
  REQUIRE(okay);

  xmlNodePtr root = doc.getRoot();

  xmlNodePtr jas1 = xmlFirstElementChild(root);

  // cusp = -0.25
  // r_ee = 3.42050023755
  PadeJastrowBuilder jastrow(elec_, psi, ptcl.getPool());
  jastrow.put(jas1);

  //target->update();
  double logpsi = psi.evaluateLog(elec_);
  REQUIRE(logpsi == -1.862821769493147);

}
}
