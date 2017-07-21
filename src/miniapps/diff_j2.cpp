//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: 
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/*!
 @file diff_j2.cpp
 @brief Two body Jastrow driver
 */

/*!
 \page miniqmc_jastrow Jastrow Factors

  The Jastrow factor accounts for correlation between electrons and ions (one-body Jastrow),
  and between electrons (two-body Jastrow).

  The Jastrow factor is composed from two types of classes - the first is for the types of
  particles involved (one or two body), and the second is the functional form for the radial part.
  The classes for the first part are qmcplusplus::OneBodyJastrowOrbital and qmcplusplus::J2OrbitalRef.
  The second part uses B-splines and is defined in qmcplusplus::BsplineFunctor.

  This miniapp only contains the B-spline functional form, since it is the most widely used.
  The QMCPACK distribution contains other functional forms.
 */


#include <Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/DistanceTable.h>
#include <OhmmsSoA/VectorSoaContainer.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <miniapps/graphite.hpp>
#include <miniapps/pseudo.hpp>
#include <Utilities/Timer.h>
#include <miniapps/common.hpp>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/J2OrbitalRef.h>
#include <QMCWaveFunctions/Jastrow/J2OrbitalSoA.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

int main(int argc, char** argv)
{

  OhmmsInfo("j2debuglogfile");

  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::ParticleLayout_t LatticeType;
  typedef ParticleSet::TensorType       TensorType;
  typedef ParticleSet::PosType          PosType;

  //use the global generator

  int na=4;
  int nb=4;
  int nc=1;
  int nsteps=100;
  int iseed=11;
  RealType Rmax(1.7);

  char *g_opt_arg;
  int opt;
  while((opt = getopt(argc, argv, "hg:i:r:")) != -1)
  {
    switch(opt)
    {
      case 'h':
        printf("[-g \"n0 n1 n2\"]\n");
        return 1;
      case 'g': //tiling1 tiling2 tiling3
        sscanf(optarg,"%d %d %d",&na,&nb,&nc);
        break;
      case 'i': //number of MC steps
        nsteps=atoi(optarg);
        break;
      case 's'://random seed
        iseed=atoi(optarg);
        break;
      case 'r'://rmax
        Rmax=atof(optarg);
        break;
    }
  }

//  Random.init(0,1,iseed);
  Tensor<int,3> tmat(na,0,0,0,nb,0,0,0,nc);

  //turn off output
  if(omp_get_max_threads()>1)
  {
    OhmmsInfo::Log->turnoff();
    OhmmsInfo::Warn->turnoff();
  }

  int nptcl=0;
  double t0=0.0,t1=0.0;
  OHMMS_PRECISION ratio=0.0;

  //list of accumulated errors
  double evaluateLog_v_err=0.0;
  double evaluateLog_g_err=0.0;
  double evaluateLog_l_err=0.0;
  double evalGrad_g_err=0.0;
  double ratioGrad_r_err=0.0;
  double ratioGrad_g_err=0.0;
  double evaluateGL_g_err=0.0;
  double evaluateGL_l_err=0.0;
  double ratio_err=0.0;

  PrimeNumberSet<uint32_t> myPrimes;

  #pragma omp parallel reduction(+:t0,ratio) \
   reduction(+:evaluateLog_v_err,evaluateLog_g_err,evaluateLog_l_err,evalGrad_g_err) \
   reduction(+:ratioGrad_r_err,ratioGrad_g_err,evaluateGL_g_err,evaluateGL_l_err,ratio_err)
  {
    ParticleSet ions, els;
    OHMMS_PRECISION scale=1.0;

    int np=omp_get_num_threads();
    int ip=omp_get_thread_num();

    //create generator within the thread
    RandomGenerator<RealType> random_th(myPrimes[ip]);

    tile_graphite(ions,tmat,scale);
    ions.RSoA=ions.R;

    const int nions=ions.getTotalNum();
    const int nels=4*nions;
    const int nels3=3*nels;

    #pragma omp master
    nptcl=nels;

    {//create up/down electrons
      els.Lattice.BoxBConds=1;   els.Lattice.set(ions.Lattice);
      vector<int> ud(2); ud[0]=nels/2; ud[1]=nels-ud[0];
      els.create(ud);
      els.R.InUnit=1;
      random_th.generate_uniform(&els.R[0][0],nels3);
      els.convert2Cart(els.R); // convert to Cartiesian
      els.RSoA=els.R;
    }

    ParticleSet els_ref(els);
    els_ref.RSoA=els_ref.R;

    //create tables
    DistanceTableData* d_ee=DistanceTable::add(els,DT_SOA);
    DistanceTableData* d_ee_ref=DistanceTable::add(els_ref,DT_SOA);

    ParticlePos_t delta(nels);

    RealType sqrttau=2.0;
    RealType accept=0.5;

    vector<RealType> ur(nels);
    random_th.generate_uniform(ur.data(),nels);

    J2OrbitalSoA<BsplineFunctor<RealType> > J(els);
    J2OrbitalRef<BsplineFunctor<RealType> > J_ref(els_ref);

    RealType r2_cut=std::min(RealType(6.4),els.Lattice.WignerSeitzRadius);
    buildJ2(J,r2_cut);
    cout << "Done with the J2 " << endl;
    buildJ2(J_ref,r2_cut);
    cout << "Done with the J2_ref " << endl;

    constexpr RealType czero(0);

    //compute distance tables
    els.update();
    els_ref.update();

    //for(int mc=0; mc<nsteps; ++mc)
    {
      els.G=czero;
      els.L=czero;
      J.evaluateLog(els,els.G,els.L);

      els_ref.G=czero;
      els_ref.L=czero;
      J_ref.evaluateLog(els_ref,els_ref.G,els_ref.L);

      cout << "Check values " << J.LogValue << " " << els.G[0] << " " << els.L[0] << endl;
      cout << "Check values ref " << J_ref.LogValue << " " << els_ref.G[0] << " " << els_ref.L[0] << endl << endl;
      cout << "evaluateLog::V Error = " << (J.LogValue-J_ref.LogValue)/nels<< endl;
      evaluateLog_v_err+=std::fabs((J.LogValue-J_ref.LogValue)/nels);
      {
        double g_err=0.0;
        for(int iel=0; iel<nels; ++iel)
        {
          PosType dr= (els.G[iel]-els_ref.G[iel]);
          RealType d=sqrt(dot(dr,dr));
          g_err += d;
        }
        cout << "evaluateLog::G Error = " << g_err/nels << endl;
        evaluateLog_g_err+=std::fabs(g_err/nels);
      }
      {
        double l_err=0.0;
        for(int iel=0; iel<nels; ++iel)
        {
          l_err += abs(els.L[iel]-els_ref.L[iel]);
        }
        cout << "evaluateLog::L Error = " << l_err/nels << endl;
        evaluateLog_l_err+=std::fabs(l_err/nels);
      }

      random_th.generate_normal(&delta[0][0],nels3);
      double g_eval=0.0;
      double r_ratio=0.0;
      double g_ratio=0.0;

      els.update();
      els_ref.update();

      int naccepted=0;

      for(int iel=0; iel<nels; ++iel)
      {
        els.setActive(iel);
        PosType grad_soa=J.evalGrad(els,iel);

        els_ref.setActive(iel);
        PosType grad_ref=J_ref.evalGrad(els_ref,iel)-grad_soa;
        g_eval+=sqrt(dot(grad_ref,grad_ref));

        PosType dr=sqrttau*delta[iel];
        bool good_soa=els.makeMoveAndCheck(iel,dr); 
        bool good_ref=els_ref.makeMoveAndCheck(iel,dr); 

        if(!good_soa) continue;

        grad_soa=0;
        RealType r_soa=J.ratioGrad(els,iel,grad_soa);
        grad_ref=0;
        RealType r_ref=J_ref.ratioGrad(els_ref,iel,grad_ref);

        grad_ref-=grad_soa;
        g_ratio+=sqrt(dot(grad_ref,grad_ref));
        r_ratio += abs(r_soa/r_ref-1);

        if(Random() < r_ref)
        {
          J.acceptMove(els,iel);
          els.acceptMove(iel);

          J_ref.acceptMove(els_ref,iel);
          els_ref.acceptMove(iel);

          naccepted++;
        }
        else
        {
          els.rejectMove(iel);
          els_ref.rejectMove(iel);
        }
      }
      cout << "Accepted " << naccepted << "/" << nels << endl;
      cout << "evalGrad::G      Error = " << g_eval/nels << endl;
      cout << "ratioGrad::G     Error = " << g_ratio/nels << endl;
      cout << "ratioGrad::Ratio Error = " << r_ratio/nels << endl;
      evalGrad_g_err+=std::fabs(g_eval/nels);
      ratioGrad_g_err+=std::fabs(g_ratio/nels);
      ratioGrad_r_err+=std::fabs(r_ratio/nels);

      //nothing to do with J2 but needs for general cases
      els.donePbyP();
      els_ref.donePbyP();

      els.G=czero;
      els.L=czero;
      J.evaluateGL(els, els.G, els.L);

      els_ref.G=czero;
      els_ref.L=czero;
      J_ref.evaluateGL(els, els_ref.G, els_ref.L);

      {
        double g_err=0.0;
        for(int iel=0; iel<nels; ++iel)
        {
          PosType dr= (els.G[iel]-els_ref.G[iel]);
          RealType d=sqrt(dot(dr,dr));
          g_err += d;
        }
        cout << "evaluteGL::G Error = " << g_err/nels << endl;
        evaluateGL_g_err+=std::fabs(g_err/nels);
      }
      {
        double l_err=0.0;
        for(int iel=0; iel<nels; ++iel)
        {
          l_err += abs(els.L[iel]-els_ref.L[iel]);
        }
        cout << "evaluteGL::L Error = " << l_err/nels << endl;
        evaluateGL_l_err+=std::fabs(l_err/nels);
      }

      //now ratio only
      r_ratio=0.0;
      constexpr int nknots=12;
      for(int iel=0; iel<nels; ++iel)
      {
        random_th.generate_uniform(&delta[0][0],nknots*3);
        for(int k=0; k<nknots;++k)
        {
          els.makeMoveOnSphere(iel,delta[k]);
          RealType r_soa=J.ratio(els,iel);
          els.rejectMove(iel);

          els_ref.makeMoveOnSphere(iel,delta[k]);
          RealType r_ref=J_ref.ratio(els_ref,iel);
          els_ref.rejectMove(iel);
          r_ratio += abs(r_soa/r_ref-1);
        }
      }
      cout << "ratio with SphereMove  Error = " << r_ratio/(nels*nknots) << endl;
      ratio_err+=std::fabs(r_ratio/(nels*nknots));
    }
  } //end of omp parallel

  int np=omp_get_max_threads();
  constexpr RealType small=std::numeric_limits<RealType>::epsilon()*1e4;
  bool fail=false;
  cout << std::endl;
  if ( evaluateLog_v_err/np > small )
  {
    cout << "Fail in evaluateLog, V error =" << evaluateLog_v_err/np << std::endl;
    fail = true;
  }
  if ( evaluateLog_g_err/np > small )
  {
    cout << "Fail in evaluateLog, G error =" << evaluateLog_g_err/np << std::endl;
    fail = true;
  }
  if ( evaluateLog_l_err/np > small )
  {
    cout << "Fail in evaluateLog, L error =" << evaluateLog_l_err/np << std::endl;
    fail = true;
  }
  if ( evalGrad_g_err/np > small )
  {
    cout << "Fail in evalGrad, G error =" << evalGrad_g_err/np << std::endl;
    fail = true;
  }
  if ( ratioGrad_r_err/np > small )
  {
    cout << "Fail in ratioGrad, ratio error =" << ratioGrad_r_err/np << std::endl;
    fail = true;
  }
  if ( ratioGrad_g_err/np > small )
  {
    cout << "Fail in ratioGrad, G error =" << ratioGrad_g_err/np << std::endl;
    fail = true;
  }
  if ( evaluateGL_g_err/np > small )
  {
    cout << "Fail in evaluateGL, G error =" << evaluateGL_g_err/np << std::endl;
    fail = true;
  }
  if ( evaluateGL_l_err/np > small )
  {
    cout << "Fail in evaluateGL, L error =" << evaluateGL_l_err/np << std::endl;
    fail = true;
  }
  if ( ratio_err/np > small )
  {
    cout << "Fail in ratio, ratio error =" << ratio_err/np << std::endl;
    fail = true;
  }
  if(!fail) cout << "All checking pass!" << std::endl;

  return 0;
}
