////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <vector>
#include "Utilities/Configuration.h"
#include "Drivers/Crowd.hpp"

#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet_builder.h"
#include "Particle/ParticleSet_builder.hpp"
#include "Particle/ParticleSet.h"

namespace qmcplusplus
{

template<>
void Crowd<Devices::CUDA>::calcNLPP(int nions, QMCT::RealType Rmax)
{
  std::cout << "Crowd CUDA calcNLPP Called \n"; 
  using T = double;
  using QPT = QMCFutureTypes<T>;
  std::vector<QPT::FuturePos> pos;
  EinsplineSPO<Devices::CUDA, T>& spo_cast = *(static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos.begin()[0]));
  auto& spline = *(spo_cast.einspline_spo_device.getDeviceEinsplines()).get();
  auto& esp = spo_cast.getParams();
  std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), spos.begin(), elss_begin(), nlpps_begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), spos.end(), elss_end(), nlpps_end())),
                [nions, Rmax, &pos, &spline, &esp](const boost::tuple<WaveFunction&, SPOSet*, ParticleSet&, NonLocalPP<QMCT::RealType>&>& t) {
                  auto& wavefunction = t.get<0>();
                  auto& spo          = *(t.get<1>());
                  auto& els          = t.get<2>();
                  auto& ecp          = t.get<3>();
                  int nknots         = ecp.size();
                  ParticleSet::ParticlePos_t rOnSphere(nknots);
                  ecp.randomize(rOnSphere); // pick random sphere
                  const DistanceTableData* d_ie = els.DistTables[wavefunction.get_ei_TableID()];

                  for (int jel = 0; jel < els.getTotalNum(); ++jel)
                  {
                    const auto& dist  = d_ie->Distances[jel];
                    const auto& displ = d_ie->Displacements[jel];
                    for (int iat = 0; iat < nions; ++iat)
                      if (dist[iat] < Rmax)
                        for (int k = 0; k < nknots; k++)
                        {
                          QMCT::PosType deltar(dist[iat] * rOnSphere[k] - displ[iat]);
			  QMCT::PosType p= els.R[jel] + deltar; 
			  pos.push_back({p[0],p[1],p[2]});
                        }
                  }
                });

  GPUArray<T,1,1> dev_v;
  std::cout << "Evaluating V for nlpp times: " << pos.size() << '\n';
  // should have a try or better yet consult a model
  int pos_block_size = 1024;
  int num_pos_blocks = pos.size() / pos_block_size;
  if (pos.size() % pos_block_size) ++num_pos_blocks;
  dev_v.resize(esp.nBlocks * num_pos_blocks, esp.nSplinesPerBlock);
  auto pos_iter = pos.begin();
  for (int ib = 0; ib < num_pos_blocks; ib++)
    {
      std::vector<QPT::FuturePos> these_pos(pos_block_size);
      for( int ip = 0; ip < pos_block_size; +ip)
	{
	  if(pos_iter != pos.end())
	    these_pos[ip] = *(pos_iter++);
	  else break;
	}
      MultiBsplineFuncs<Devices::CUDA, T> compute_engine;
      compute_engine.evaluate_v(spline.get()[0],
				these_pos,
				dev_v.get_devptr(),
				esp.nBlocks,
				esp.nSplines,
				esp.nSplinesPerBlock);
    }
      //This likely needs to temporary update to els.
  //In the spirit of miniqmc I'm just going to run it the correct numnber of times.
  auto& els = *(*(elss.begin()));
  for(int i = 0; i < pos.size(); ++i)
    wavefunctions[0]->ratio(els, 1);

}

template<>
void Crowd<Devices::CUDA>::evaluateHessian(int iel)
{
  
  using T = double;
  //xo for now this does use a singple EinsplineSPO to call through
  std::vector<EinsplineSPODeviceImp<Devices::CUDA, T>::HessianParticipants> hessian_participants; //(pack_size_);
  std::vector<std::array<T, 3>> pos(pack_size_);
  // Better be the same for all
  EinsplineSPO<Devices::CUDA, T>& spo = static_cast<EinsplineSPO<Devices::CUDA, T>&>(*(spos[0]));
  EinsplineSPOParams<T> esp = spo.getParams();
  for(int i=0; i < pack_size_; ++i)
    {
      // gross
      hessian_participants.push_back(static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos[i])->einspline_spo_device.visit_for_vgh());
      auto r = elss[i]->R[iel];
      auto u = esp.lattice.toUnit_floor(r);
      pos[i] = {u[0], u[1], u[2]};
    }
  
  GPUArray<T, 1, 1> dev_psi;
  GPUArray<T, 3, 1> dev_grad;
  GPUArray<T, 6, 1> dev_hess;
  //Should be in try block for memory allocation failure
  //Possible async opportunity
  dev_psi.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
  dev_hess.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
  dev_grad.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
  MultiBsplineFuncs<Devices::CUDA, T> compute_engine;
  const auto& spline = static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos[0])->einspline_spo_device.getDeviceEinsplines();
  compute_engine.evaluate_vgh((*spline)[0],
                              pos,
                              dev_psi.get_devptr(),
                              dev_grad.get_devptr(),
                              dev_hess.get_devptr(),
			      esp.nBlocks,
			      esp.nSplines,
                              esp.nSplinesPerBlock);

  //Now we have to copy the data back to the participants
  for(int i= 0 ; i < pack_size_; ++i)
    {
      dev_psi.pull(hessian_participants[i].psi, i*esp.nBlocks, esp.nBlocks);
      dev_grad.pull(hessian_participants[i].grad, i*esp.nBlocks, esp.nBlocks);
      dev_hess.pull(hessian_participants[i].hess, i*esp.nBlocks, esp.nBlocks);
    }
      
  
}
  
} // namespace qmcplusplus

#include "Drivers/CrowdCUDA.hpp"

namespace qmcplusplus
{
template class Crowd<Devices::CUDA>;
}
