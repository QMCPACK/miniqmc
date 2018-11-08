#include "QMCWaveFunctions/future/DeterminantDeviceImp.h"
#include "QMCWaveFunctions/future/DeterminantDevice.h"
namespace qmcplusplus
{
namespace future
{
template class DeterminantDeviceImp<DDT::CPU>;
#ifdef QMC_USE_KOKKOS
template class DeterminantDeviceImp<DDT::KOKKOS>;
#endif

}
}
