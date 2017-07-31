//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_wfc.cpp
 * @brief Miniapp to check individual wave function component against its reference.
 */

#include <Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/DistanceTable.h>
#include <OhmmsSoA/VectorSoaContainer.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <miniapps/nio.hpp>
#include <miniapps/pseudo.hpp>
#include <Utilities/Timer.h>
#include <miniapps/common.hpp>
#include <QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h>
#include <QMCWaveFunctions/Jastrow/JeeIOrbitalRef.h>
#include <QMCWaveFunctions/Jastrow/JeeIOrbitalSoA.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/J1OrbitalRef.h>
#include <QMCWaveFunctions/Jastrow/J1OrbitalSoA.h>
#include <QMCWaveFunctions/Jastrow/J2OrbitalRef.h>
#include <QMCWaveFunctions/Jastrow/J2OrbitalSoA.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

int main(int argc, char** argv)
{

  OhmmsInfo("CheckWaveFunctionComponents");

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
  string wfc_name("J2");

  char *g_opt_arg;
  int opt;
  while((opt = getopt(argc, argv, "hg:i:r:f:")) != -1)
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
      case 'f':// Wave function component
        wfc_name=optarg;
        break;
    }
  }

  if(wfc_name!="J1"&&wfc_name!="J2"&&wfc_name!="JeeI")
  {
    cerr << "Uknown wave funciton component " << wfc_name << endl
         << "Now supports J1 J2(default) JeeI Det(to be supported)" << endl;
    exit(1);
  }

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
    ions.setName("ion");
    els.setName("e");
    OHMMS_PRECISION scale=1.0;

    int np=omp_get_num_threads();
    int ip=omp_get_thread_num();

    //create generator within the thread
    RandomGenerator<RealType> random_th(myPrimes[ip]);

    tile_graphite(ions,tmat,scale);
    ions.RSoA=ions.R; //fill the SoA

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
    DistanceTableData* d_ie=DistanceTable::add(ions,els_ref,DT_SOA);
    d_ie->setRmax(Rmax);

    ParticlePos_t delta(nels);

    RealType sqrttau=2.0;
    RealType accept=0.5;

    vector<RealType> ur(nels);
    random_th.generate_uniform(ur.data(),nels);

    OrbitalBasePtr wfc=nullptr;
    OrbitalBasePtr wfc_ref=nullptr;
    if(wfc_name=="J2")
    {
      RealType r2_cut=std::min(RealType(6.4),els.Lattice.WignerSeitzRadius);
      J2OrbitalSoA<BsplineFunctor<RealType> >* J=new J2OrbitalSoA<BsplineFunctor<RealType> >(els);
      buildJ2(*J,r2_cut);
      wfc=dynamic_cast<OrbitalBasePtr>(J);
      cout << "Built J2" << endl;
      J2OrbitalRef<BsplineFunctor<RealType> >* J_ref=new J2OrbitalRef<BsplineFunctor<RealType> >(els_ref);
      buildJ2(*J_ref,r2_cut);
      wfc_ref=dynamic_cast<OrbitalBasePtr>(J_ref);
      cout << "Built J2_ref" << endl;
    }
    else if(wfc_name=="J1")
    {
      // FIXME right way to add rcut for Ni = 4.8261684030, O = 4.8261684030 ?
      RealType r1_cut=std::min(RealType(6.4),els.Lattice.WignerSeitzRadius);
      J1OrbitalSoA<BsplineFunctor<RealType> >* J=new J1OrbitalSoA<BsplineFunctor<RealType> >(ions,els);
      buildJ1(*J,r1_cut);
      wfc=dynamic_cast<OrbitalBasePtr>(J);
      cout << "Built J1" << endl;
      J1OrbitalRef<BsplineFunctor<RealType> >* J_ref=new J1OrbitalRef<BsplineFunctor<RealType> >(ions,els_ref);
      buildJ1(*J_ref,r1_cut);
      wfc_ref=dynamic_cast<OrbitalBasePtr>(J_ref);
      cout << "Built J1_ref" << endl;
    }
    else if(wfc_name=="JeeI")
    {
      RealType r_cut=std::min(RealType(6.0),els.Lattice.WignerSeitzRadius);
      JeeIOrbitalSoA<PolynomialFunctor3D>* J=new JeeIOrbitalSoA<PolynomialFunctor3D>(ions,els);
      buildJeeI(*J,r_cut);
      wfc=dynamic_cast<OrbitalBasePtr>(J);
      cout << "Built JeeI" << endl;
      JeeIOrbitalRef<PolynomialFunctor3D>* J_ref=new JeeIOrbitalRef<PolynomialFunctor3D>(ions,els_ref);
      buildJeeI(*J_ref,r_cut);
      wfc_ref=dynamic_cast<OrbitalBasePtr>(J_ref);
      cout << "Built JeeI_ref" << endl;
    }

    constexpr RealType czero(0);

    //compute distance tables
    els.update();
    els_ref.update();

    //for(int mc=0; mc<nsteps; ++mc)
    {
      els.G=czero;
      els.L=czero;
      wfc->evaluateLog(els,els.G,els.L);

      els_ref.G=czero;
      els_ref.L=czero;
      wfc_ref->evaluateLog(els_ref,els_ref.G,els_ref.L);

      cout << "Check values " << wfc->LogValue << " " << els.G[12] << " " << els.L[12] << endl;
      cout << "Check values ref " << wfc_ref->LogValue << " " << els_ref.G[12] << " " << els_ref.L[12] << endl << endl;
      cout << "evaluateLog::V Error = " << (wfc->LogValue-wfc_ref->LogValue)/nels<< endl;
      evaluateLog_v_err+=std::fabs((wfc->LogValue-wfc_ref->LogValue)/nels);
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

      int naccepted=0;

      for(int iel=0; iel<nels; ++iel)
      {
        els.setActive(iel);
        PosType grad_soa=wfc->evalGrad(els,iel);

        els_ref.setActive(iel);
        PosType grad_ref=wfc_ref->evalGrad(els_ref,iel)-grad_soa;
        g_eval+=sqrt(dot(grad_ref,grad_ref));

        PosType dr=sqrttau*delta[iel];
        bool good_soa=els.makeMoveAndCheck(iel,dr); 
        bool good_ref=els_ref.makeMoveAndCheck(iel,dr); 

        if(!good_ref) continue;

        grad_soa=0;
        RealType r_soa=wfc->ratioGrad(els,iel,grad_soa);
        grad_ref=0;
        RealType r_ref=wfc_ref->ratioGrad(els_ref,iel,grad_ref);

        grad_ref-=grad_soa;
        g_ratio+=sqrt(dot(grad_ref,grad_ref));
        r_ratio += abs(r_soa/r_ref-1);

        if(ur[iel] < r_ref)
        {
          wfc->acceptMove(els,iel);
          els.acceptMove(iel);

          wfc_ref->acceptMove(els_ref,iel);
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
      wfc->evaluateGL(els, els.G, els.L);

      els_ref.G=czero;
      els_ref.L=czero;
      wfc_ref->evaluateGL(els_ref,els_ref.G,els_ref.L);

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
      int nsphere=0;
      for(int iat=0; iat<nions; ++iat)
      {
        for(int nj=0, jmax=d_ie->nadj(iat); nj<jmax; ++nj)
        {
          const RealType r=d_ie->distance(iat,nj);
          if(r<Rmax)
          {
            const int iel=d_ie->iadj(iat,nj);
            nsphere++;
            random_th.generate_uniform(&delta[0][0],nknots*3);
            for(int k=0; k<nknots;++k)
            {
              els.makeMoveOnSphere(iel,delta[k]);
              RealType r_soa=wfc->ratio(els,iel);
              els.rejectMove(iel);

              els_ref.makeMoveOnSphere(iel,delta[k]);
              RealType r_ref=wfc_ref->ratio(els_ref,iel);
              els_ref.rejectMove(iel);
              r_ratio += abs(r_soa/r_ref-1);
            }
          }
        }
      }
      cout << "ratio with SphereMove  Error = " << r_ratio/nsphere << " # of moves =" << nsphere << endl;
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
