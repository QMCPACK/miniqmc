#include <iostream>
#include "OMPVector.h"
#include "OMPTinyVector.h"
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

  // OMPTinyVector
  OMPVector<OMPTinyVector<float, 3> > my_tinyvec(len);
  for(size_t i=0; i<len; i++)
    my_tinyvec.vec[i] = i;
  OMPTinyVector<float, 3> *tinyvec_ptr = my_tinyvec.data();
  my_tinyvec.update_to_device();
  #pragma omp target teams distribute parallel for
  for(size_t i=0; i<len; i++)
    tinyvec_ptr[i][0] = tinyvec_ptr[i][0] * 2;
  my_tinyvec.update_from_device();
  // print value
  for(size_t i=0; i<len; i++)
    std::cout << "i=" << i << " value=" << my_tinyvec.vec[i] << std::endl;
}
