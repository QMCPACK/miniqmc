
#ifndef QMCPLUSPLUS_CROWD_BUFFERS_HPP
#define QMCPLUSPLUS_CROWD_BUFFERS_HPP

namespace qmcplusplus
{
template<Devices DT>
class CrowdBuffers
{

};

#ifdef QMC_USE_CUDA
#include "CUDA/GPUArray.h"
template<>
class CrowdBuffers<Devices::CUDA>
{
public:
  using T = double;
  T* psi;
  T* grad;
  T* hess;
  GPUArray<T, 1, 1> dev_psi;
  GPUArray<T, 3, 1> dev_grad;
  GPUArray<T, 6, 1> dev_hess;
};
#endif
}

#endif
