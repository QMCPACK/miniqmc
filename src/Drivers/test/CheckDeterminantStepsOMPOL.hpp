#include "Drivers/check_determinant.h"

namespace qmcplusplus
{

template<>
void CheckDeterminantSteps<Devices::OMPOL>::updateFromDevice(DiracDeterminant<DT>& determinant_device)
{
  determinant_device.transfer_from_device();
}

template<>
CheckDeterminantSteps<DDT:OMPOL>::runThreads(int np,
						       PrimeNumberSet<uint32_t>& myPrimes,
			  ParticleSet& ions, int& nsteps,
						       int& nsubsteps)
{
  double accumulated_error = 0.0;
#pragma omp parallel reduction(+ : accumulated_error)
  {
    accumulated_error += this->thread_main(PrimeNumberSet<uint32_t>& myPrimes,
		      ParticleSet& ions, int& nsteps,
		      int& nsubsteps)

    // accumulate error
    for (int i = 0; i < determinant_ref.size(); i++)
    {
      accumulated_error += std::fabs(determinant_ref(i) - determinant(i));
    }
  } // end of omp parallel

  return accumulated_error;
}

}
