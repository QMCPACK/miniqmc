#include "Devices.h"
#include "QMCWaveFunctions/EinsplineSPODeviceImpCPU.hpp"

namespace qmcplusplus
{
  template class EinsplineSPODeviceImp<Devices::CPU, float>;
  template class EinsplineSPODeviceImp<Devices::CPU, double>;
}
