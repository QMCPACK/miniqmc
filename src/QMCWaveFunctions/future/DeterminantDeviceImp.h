
#ifndef QMCPLUSPLUS_FUTURE_DETERMINANT_DEVICE_IMP_H
#define QMCPLUSPLUS_FUTURE_DETERMINANT_DEVICE_IMP_H

namespace qmcplusplus
{
namespace future
{

enum DeterminantDeviceType
{
  KOKKOS,
  OMPOL
}

template<DeterminantDeviceType DT, Batching B>
class DeterminantDeviceTypeImp;

}
}

#endif
