#include "catch.hpp"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"

namespace qmcpluplus
{

TEST_CASE("EinsplineSPODevice Instantiation", "[wavefunction]")
{
  EinsplineSPODevice<Device::CUDA,double> cuda_ein;
}
}
