#ifndef QMCPLUSPLUS_LINALG_CUDA_H
#define QMCPLUSPLUS_LINALG_CUDA_H

namespace qmcplusplus
{
struct LinAlgCUDA : public LinAlgCPU
{
  // most of the calls in the CPU version exist in the magma library in batched and unbatched forms. 
}

} // namespace qmcplusplus
#endif
