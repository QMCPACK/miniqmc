#include "Devices.h"
#include "QMCWaveFunctions/EinsplineSPODeviceImpKOKKOS.hpp"

namespace qmcplusplus
{
  template class EinsplineSPODeviceImp<Devices::KOKKOS, double>;
  template class EinsplineSPODeviceImp<Devices::KOKKOS, float>;
}
