#include <iostream>
#include "OMPVector.h"
#include <omp.h>

using namespace qmcplusplus;

int main()
{
  const int len = 12;
  // set intial value
  OMPVector<int> myvec(len);
  for(size_t i=0; i<len; i++)
    myvec.vec[i] = i;
  int *vec_ptr = myvec.data();
  std::cout << "Current number of devices " << omp_get_num_devices() << std::endl;
  std::cout << "mapped already? " << omp_target_is_present(vec_ptr,0) << std::endl;
  myvec.update_to_device();
  #pragma omp target teams distribute parallel for
  for(size_t i=0; i<len; i++)
    vec_ptr[i] = vec_ptr[i] * 2;
  myvec.update_from_device();
  // print value
  for(size_t i=0; i<len; i++)
    std::cout << "i=" << i << " value=" << myvec.vec[i] << std::endl;
}
