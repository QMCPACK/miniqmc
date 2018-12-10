#include "Devices.h"
#include "QMCWaveFunctions/EinsplineSPODeviceImpCPU.hpp"

namespace qmcplusplus
{
  template class EinsplineSPODeviceImp<Devices::CPU, float>;
  template class EinsplineSPODeviceImp<Devices::CPU, double>;
}
template class std::vector<multi_UBspline_3d_d<qmcplusplus::Devices::CPU>*, qmcplusplus::Mallocator<multi_UBspline_3d_d<(qmcplusplus::Devices)0>*, 32ul> >;
template class std::vector<qmcplusplus::VectorSoAContainer<double, 6u>, qmcplusplus::Mallocator<qmcplusplus::VectorSoAContainer<double, 6u>, 32ul> >;
template class std::vector<std::vector<double, qmcplusplus::Mallocator<double, 32ul>>, qmcplusplus::Mallocator<std::vector<double, qmcplusplus::Mallocator<double, 32ul>>,32ul>>;
template class std::vector<double, qmcplusplus::Mallocator<double, 32ul>>;

