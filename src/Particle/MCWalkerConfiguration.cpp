//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jordan E. Vincent, University of Illinois at Urbana-Champaign
//                    Bryan Clark, bclark@Princeton.edu, Princeton University
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Cynthia Gu, zg1@ornl.gov, Oak Ridge National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    



#include "Utilities/OhmmsInfo.h"
#include "Particle/MCWalkerConfiguration.h"
#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "Utilities/IteratorUtility.h"
#include <map>

#ifdef QMC_CUDA
#include "Particle/accept_kernel.h"
#endif

namespace qmcplusplus
{

MCWalkerConfiguration::MCWalkerConfiguration():
  OwnWalkers(true),ReadyForPbyP(false),UpdateMode(Update_Walker),

  MaxSamples(10),CurSampleCount(0),GlobalNumWalkers(0)
#ifdef QMC_CUDA
  ,RList_GPU("MCWalkerConfiguration::RList_GPU"),
  GradList_GPU("MCWalkerConfiguration::GradList_GPU"),
  LapList_GPU("MCWalkerConfiguration::LapList_GPU"),
  Rnew_GPU("MCWalkerConfiguration::Rnew_GPU"),
  NLlist_GPU ("MCWalkerConfiguration::NLlist_GPU"),
  AcceptList_GPU("MCWalkerConfiguration::AcceptList_GPU"),
  iatList_GPU("iatList_GPU")
#endif
{
  //move to ParticleSet
  //initPropertyList();
}

MCWalkerConfiguration::MCWalkerConfiguration(const MCWalkerConfiguration& mcw)
  : ParticleSet(mcw), OwnWalkers(true), GlobalNumWalkers(mcw.GlobalNumWalkers),
    UpdateMode(Update_Walker), ReadyForPbyP(false),
    MaxSamples(mcw.MaxSamples), CurSampleCount(0)
#ifdef QMC_CUDA
    ,RList_GPU("MCWalkerConfiguration::RList_GPU"),
    GradList_GPU("MCWalkerConfiguration::GradList_GPU"),
    LapList_GPU("MCWalkerConfiguration::LapList_GPU"),
    Rnew_GPU("MCWalkerConfiguration::Rnew_GPU"),
    NLlist_GPU ("MCWalkerConfiguration::NLlist_GPU"),
    AcceptList_GPU("MCWalkerConfiguration::AcceptList_GPU"),
    iatList_GPU("iatList_GPU")
#endif
{
  GlobalNumWalkers=mcw.GlobalNumWalkers;
  WalkerOffsets=mcw.WalkerOffsets;
  Properties=mcw.Properties;
  //initPropertyList();
}

///default destructor
MCWalkerConfiguration::~MCWalkerConfiguration()
{
  if(OwnWalkers)
    destroyWalkers(WalkerList.begin(), WalkerList.end());
}


void MCWalkerConfiguration::createWalkers(int n)
{
  if(WalkerList.empty())
  {
    while(n)
    {
      Walker_t* awalker=new Walker_t(TotalNum);
      awalker->R = R;
      WalkerList.push_back(awalker);
      --n;
    }
  }
  else
  {
    if(WalkerList.size()>=n)
    {
      int iw=WalkerList.size();//copy from the back
      for(int i=0; i<n; ++i)
      {
        WalkerList.push_back(new Walker_t(*WalkerList[--iw]));
      }
    }
    else
    {
      int nc=n/WalkerList.size();
      int nw0=WalkerList.size();
      for(int iw=0; iw<nw0; ++iw)
      {
        for(int ic=0; ic<nc; ++ic)
          WalkerList.push_back(new Walker_t(*WalkerList[iw]));
      }
      n-=nc*nw0;
      while(n>0)
      {
        WalkerList.push_back(new Walker_t(*WalkerList[--nw0]));
        --n;
      }
    }
  }
  resizeWalkerHistories();
}


void MCWalkerConfiguration::resize(int numWalkers, int numPtcls)
{
  if(TotalNum && WalkerList.size())
    app_warning() << "MCWalkerConfiguration::resize cleans up the walker list." << std::endl;
  ParticleSet::resize(numPtcls);
  int dn=numWalkers-WalkerList.size();
  if(dn>0)
    createWalkers(dn);
  if(dn<0)
  {
    int nw=-dn;
    if(nw<WalkerList.size())
    {
      iterator it = WalkerList.begin();
      while(nw)
      {
        delete *it;
        ++it;
        --nw;
      }
      WalkerList.erase(WalkerList.begin(),WalkerList.begin()-dn);
    }
  }
  //iterator it = WalkerList.begin();
  //while(it != WalkerList.end()) {
  //  delete *it++;
  //}
  //WalkerList.erase(WalkerList.begin(),WalkerList.end());
  //R.resize(np);
  //GlobalNum = np;
  //createWalkers(nw);
}

///returns the next valid iterator
MCWalkerConfiguration::iterator
MCWalkerConfiguration::destroyWalkers(iterator first, iterator last)
{
  if(OwnWalkers)
  {
    iterator it = first;
    while(it != last)
    {
      delete *it++;
    }
  }
  return WalkerList.erase(first,last);
}

void MCWalkerConfiguration::createWalkers(iterator first, iterator last)
{
  destroyWalkers(WalkerList.begin(),WalkerList.end());
  OwnWalkers=true;
  while(first != last)
  {
    WalkerList.push_back(new Walker_t(**first));
    ++first;
  }
}

void
MCWalkerConfiguration::destroyWalkers(int nw)
{
  if(nw > WalkerList.size())
  {
    app_warning() << "  Cannot remove walkers. Current Walkers = " << WalkerList.size() << std::endl;
    return;
  }
  nw=WalkerList.size()-nw;
  int iw=nw;
  while(iw<WalkerList.size())
  {
    delete WalkerList[iw++];
  }
  //iterator it(WalkerList.begin()+nw),it_end(WalkerList.end());
  //while(it != it_end)
  //{
  //  delete *it++;
  //}
  WalkerList.erase(WalkerList.begin()+nw,WalkerList.end());
}

void MCWalkerConfiguration::copyWalkers(iterator first, iterator last, iterator it)
{
  while(first != last)
  {
    (*it++)->makeCopy(**first++);
  }
}


void
MCWalkerConfiguration::copyWalkerRefs(Walker_t* head, Walker_t* tail)
{
  if(OwnWalkers)
    //destroy the current walkers
  {
    destroyWalkers(WalkerList.begin(), WalkerList.end());
    WalkerList.clear();
    OwnWalkers=false;//set to false to prevent deleting the Walkers
  }
  if(WalkerList.size()<2)
  {
    WalkerList.push_back(0);
    WalkerList.push_back(0);
  }
  WalkerList[0]=head;
  WalkerList[1]=tail;
}

void MCWalkerConfiguration::reset()
{
  iterator it(WalkerList.begin()), it_end(WalkerList.end());
  while(it != it_end)
    //(*it)->reset();++it;}
  {
    (*it)->Weight=1.0;
    (*it)->Multiplicity=1.0;
    ++it;
  }
}

}

