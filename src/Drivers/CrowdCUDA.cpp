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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utilities/Configuration.h"
#include "Drivers/Crowd.hpp"

#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet_builder.h"
#include "Particle/ParticleSet_builder.hpp"
#include "Particle/ParticleSet.h"

namespace qmcplusplus
{

template<>
void Crowd<Devices::CUDA>::init()
{
  // This initializes the threads gpu
  Gpu& gpu = Gpu::get();
}
    
template<>
void Crowd<Devices::CUDA>::calcNLPP(int nions, QMCT::RealType Rmax)
{
    //std::cout << "Crowd CUDA calcNLPP Called \n";
  using T   = double;
  using QPT = QMCFutureTypes<T>;
  std::vector<QPT::FuturePos> pos;
  EinsplineSPO<Devices::CUDA, T>& spo_cast = *(static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos.begin()[0]));
  auto& spline                             = *(spo_cast.einspline_spo_device.getDeviceEinsplines()).get();
  auto& esp                                = spo_cast.getParams();
  std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), spos.begin(), elss_begin(), nlpps_begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), spos.end(), elss_end(), nlpps_end())),
                [nions, Rmax, &pos, &spline, &esp](
                    const boost::tuple<WaveFunction&, SPOSet*, ParticleSet&, NonLocalPP<QMCT::RealType>&>& t) {
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
                          QMCT::PosType p = els.R[jel] + deltar;
                          pos.push_back({p[0], p[1], p[2]});
                        }
                  }
                });

  
  //std::cout << "Evaluating V for nlpp times: " << pos.size() << '\n';
  // should have a try or better yet consult a model
  int pos_block_size = 256;
  int num_pos_blocks = pos.size() / pos_block_size;
  if (pos.size() % pos_block_size)
    ++num_pos_blocks;
  buffers_.dev_v_nlpp.resize(esp.nBlocks * pos_block_size * num_pos_blocks, esp.nSplinesPerBlock );
  auto pos_iter = pos.begin();
  MultiBsplineFuncs<Devices::CUDA, T> compute_engine;
  for (int ib = 0; ib < num_pos_blocks; ib++)
  {
    std::vector<QPT::FuturePos> these_pos(pos_block_size);
    for (int ip = 0; ip < pos_block_size; ++ip)
    {
      if (pos_iter != pos.end())
        these_pos[ip] = *(pos_iter++);
      else
        break;
    }
    size_t dev_offset = esp.nBlocks * esp.nSplinesPerBlock * pos_block_size * ib;
    buffers_.compute_engine
        .evaluate_v(spline.get()[0], these_pos, buffers_.dev_v_nlpp.get_devptr() + dev_offset, esp.nBlocks, esp.nSplines, esp.nSplinesPerBlock);
  }
  cudaStreamSynchronize(cudaStreamPerThread);
  //This likely needs to temporary update to els.
  //In the spirit of miniqmc I'm just going to run it the correct numnber of times.
  
  auto& els = *(*(elss.begin()));
  for (int i = 0; i < pos.size(); ++i)
    wavefunctions[0]->ratio(els, 1);
}

template<>
[[clang::xray_always_instrument]] void Crowd<Devices::CUDA>::evaluateHessian(int iel)
{
  using T = double;

  // Better be the same for all
  EinsplineSPO<Devices::CUDA, T>& spo = static_cast<EinsplineSPO<Devices::CUDA, T>&>(*(spos[0]));
  EinsplineSPOParams<T> esp           = spo.getParams();

  if(buffers_.dev_psi.getWidth() == 0)
    {
      double size_buffer = esp.nBlocks * pack_size_ * esp.nSplinesPerBlock * sizeof(T) * (1+3+6);
      double size_buffer_MB = size_buffer / 1024 / 1024;
      std::cout << "Allocating GPU buffers for  nBlocks = " << esp.nBlocks
		<< " pack_size = " << pack_size_ << " esp.nSplinesPerBlock = "
		<< esp.nSplinesPerBlock << " vgh eval of: " << size_buffer_MB << "MB\n";
      buffers_.dev_psi.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
      buffers_.dev_hess.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
      buffers_.dev_grad.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
    }
  
  //for now this does use a single EinsplineSPO to call through
  std::vector<HessianParticipants<Devices::CUDA, T>> hessian_participants; //(pack_size_);
  std::vector<std::array<T, 3>> pos(pack_size_);

  for (int i = 0; i < pack_size_; ++i)
  {
    hessian_participants.push_back(
        static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos[i])->einspline_spo_device.visit_for_vgh());
    auto r = elss[i]->R[iel];
    auto u = esp.lattice.toUnit_floor(r);
    pos[i] = {u[0], u[1], u[2]};
  }

  // its possible that launch multiple kernels per pack is better.
  int n_pos = pos.size();
  
  const auto& spline =
      static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos[0])->einspline_spo_device.getDeviceEinsplines();
  cudaStream_t stream = cudaStreamPerThread;

    buffers_.compute_engine.evaluate_vgh((*spline)[0],
				  pos,
				buffers_.dev_psi.get_devptr(),
				buffers_.dev_grad.get_devptr(),
				buffers_.dev_hess.get_devptr(),
                              esp.nBlocks,
                              esp.nSplines,
				esp.nSplinesPerBlock,
	stream);

    buffers_.psi(stream);
    buffers_.psi.resize(esp.nBlocks * esp.nSplinesPerBlock * sizeof(T) * n_pos);
    buffers_.psi.copyFromDevice(buffers_.dev_psi.get_devptr());
    buffers_.grad(stream);
    buffers_.grad.resize(esp.nBlocks * esp.nSplinesPerBlock * sizeof(T) * n_pos * 3);
    buffers_.grad.copyFromDevice(buffers_.dev_grad.get_devptr());
    buffers_.hess(stream);
    buffers_.hess.resize(esp.nBlocks * esp.nSplinesPerBlock * sizeof(T) * n_pos * 6);
    buffers_.hess.copyFromDevice(buffers_.dev_hess.get_devptr());
    
  //Now we have to copy the data back to the participants

  for (int i = 0; i < pack_size_; ++i)
  {
      for(int j = 0; j < esp.nBlocks; ++j)
      {
  	  buffers_.psi.toNormalTcpy(hessian_participants[i].psi[j].data(), i * esp.nBlocks * esp.nSplinesPerBlock + j * esp.nSplinesPerBlock, esp.nSplinesPerBlock);

  	  buffers_.grad.toNormalTcpy(hessian_participants[i].grad[j].data(), i * esp.nBlocks * esp.nSplinesPerBlock + j * esp.nSplinesPerBlock * 3, esp.nSplinesPerBlock * 3);

  	  buffers_.hess.toNormalTcpy(hessian_participants[i].hess[j].data(), i * esp.nBlocks * esp.nSplinesPerBlock + j * esp.nSplinesPerBlock * 6, esp.nSplinesPerBlock * 6);
      }
  }

}

template<>
[[clang::xray_always_instrument]] void Crowd<Devices::CUDA>::evaluateLaplacian(int iel)
{
  using T = double;

  // Better be the same for all
  EinsplineSPO<Devices::CUDA, T>& spo = static_cast<EinsplineSPO<Devices::CUDA, T>&>(*(spos[0]));
  EinsplineSPOParams<T> esp           = spo.getParams();

  if(buffers_.dev_psi.getWidth() == 0)
    {
      double size_buffer = esp.nBlocks * pack_size_ * esp.nSplinesPerBlock * sizeof(T) * (1+3+6);
      double size_buffer_MB = size_buffer / 1024 / 1024;
      std::cout << "Allocating GPU buffers for  nBlocks = " << esp.nBlocks
		<< " pack_size = " << pack_size_ << " esp.nSplinesPerBlock = "
		<< esp.nSplinesPerBlock << " vgh eval of: " << size_buffer_MB << "MB\n";
      buffers_.dev_psi.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
      buffers_.dev_hess.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
      buffers_.dev_grad.resize(esp.nBlocks * pack_size_, esp.nSplinesPerBlock);
    }
  
  //for now this does use a single EinsplineSPO to call through
  std::vector<HessianParticipants<Devices::CUDA, T>> hessian_participants; //(pack_size_);
  std::vector<std::array<T, 3>> pos(pack_size_);

  for (int i = 0; i < pack_size_; ++i)
  {
    hessian_participants.push_back(
        static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos[i])->einspline_spo_device.visit_for_vgh());
    auto r = elss[i]->R[iel];
    auto u = esp.lattice.toUnit_floor(r);
    pos[i] = {u[0], u[1], u[2]};
  }

  // its possible that launch multiple kernels per pack is better.
  int n_pos = pos.size();
  
  MultiBsplineFuncs<Devices::CUDA, T> compute_engine;
  const auto& spline =
      static_cast<EinsplineSPO<Devices::CUDA, T>*>(spos[0])->einspline_spo_device.getDeviceEinsplines();
  cudaStream_t stream = cudaStreamPerThread;

    compute_engine.evaluate_vgh((*spline)[0],
				  pos,
				buffers_.dev_psi.get_devptr(),
				buffers_.dev_grad.get_devptr(),
				buffers_.dev_hess.get_devptr(),
                              esp.nBlocks,
                              esp.nSplines,
				esp.nSplinesPerBlock,
	stream);

    buffers_.psi(stream);
    buffers_.psi.resize(esp.nBlocks * esp.nSplinesPerBlock * sizeof(T) * n_pos);
    buffers_.psi.copyFromDevice(buffers_.dev_psi.get_devptr());
    buffers_.grad(stream);
    buffers_.grad.resize(esp.nBlocks * esp.nSplinesPerBlock * sizeof(T) * n_pos * 3);
    buffers_.grad.copyFromDevice(buffers_.dev_grad.get_devptr());
    buffers_.hess(stream);
    buffers_.hess.resize(esp.nBlocks * esp.nSplinesPerBlock * sizeof(T) * n_pos * 6);
    buffers_.hess.copyFromDevice(buffers_.dev_hess.get_devptr());
    
  //Now we have to copy the data back to the participants
    cudaStreamSynchronize(cudaStreamPerThread);
  for (int i = 0; i < pack_size_; ++i)
  {
      for(int j = 0; j < esp.nBlocks; ++j)
      {
  	  buffers_.psi.toNormalTcpy(hessian_participants[i].psi[j].data(), i * esp.nBlocks * esp.nSplinesPerBlock + j * esp.nSplinesPerBlock, esp.nSplinesPerBlock);

  	  buffers_.grad.toNormalTcpy(hessian_participants[i].grad[j].data(), i * esp.nBlocks * esp.nSplinesPerBlock + j * esp.nSplinesPerBlock * 3, esp.nSplinesPerBlock * 3);

  	  buffers_.hess.toNormalTcpy(hessian_participants[i].hess[j].data(), i * esp.nBlocks * esp.nSplinesPerBlock + j * esp.nSplinesPerBlock * 6, esp.nSplinesPerBlock * 6);
      }
  }

}


  } // namespace qmcplusplus

#include "Drivers/CrowdCUDA.hpp"

  namespace qmcplusplus
  {
  template class Crowd<Devices::CUDA>;
  }
