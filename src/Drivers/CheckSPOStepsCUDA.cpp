#include "Drivers/check_spo.h"
#include "Drivers/CheckSPOStepsCUDA.hpp"

namespace qmcplusplus
{
template class CheckSPOSteps<Devices::CUDA>;
//template void CheckSPOSteps<Devices::CUDA>::initialize(int, char**);

} // namespace qmcplusplus
