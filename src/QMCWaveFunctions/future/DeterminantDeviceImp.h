
#ifndef QMCPLUSPLUS_FUTURE_DETERMINANT_DEVICE_IMP_H
#define QMCPLUSPLUS_FUTURE_DETERMINANT_DEVICE_IMP_H

namespace qmcplusplus
{
namespace future
{

enum DeterminantDeviceType
{
  CPU,
  KOKKOS,
  OMPOL
};

template<DeterminantDeviceType DT>
class DeterminantDeviceImp;

}
}

#endif
