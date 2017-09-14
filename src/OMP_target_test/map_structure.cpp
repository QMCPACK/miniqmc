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
    myvec[i] = i;
  int *vec_ptr = myvec.data();
  std::cout << "Current number of devices " << omp_get_num_devices() << std::endl;
  std::cout << "mapped already? " << omp_target_is_present(vec_ptr,0) << std::endl;
  myvec.update_to_device();
  #pragma omp target teams distribute parallel for
  for(size_t i=0; i<len; i++)
    vec_ptr[i] = vec_ptr[i] * omp_get_team_num();
  myvec.update_from_device();
  // print value
  for(size_t i=0; i<len; i++)
    std::cout << "i=" << i << " value=" << myvec[i] << std::endl;

  // OMPTinyVector
  OMPVector<OMPTinyVector<float, 3> > my_tinyvec(len);
  for(size_t i=0; i<len; i++)
    my_tinyvec[i] = i;
  OMPTinyVector<float, 3> *tinyvec_ptr = my_tinyvec.data();
  my_tinyvec.update_to_device();
  #pragma omp target teams distribute parallel for
  for(size_t i=0; i<len; i++)
    tinyvec_ptr[i][0] = tinyvec_ptr[i][0] * 2;
  my_tinyvec.update_from_device();
  // print value
  for(size_t i=0; i<len; i++)
    std::cout << "i=" << i << " value=" << my_tinyvec[i] << std::endl;

  std::vector<OMPVector<int> > vec_th;
  OMPVector<int *> shadow;
  #pragma omp parallel
  {
    #pragma omp single
    {
      const size_t nt=omp_get_num_threads();
      vec_th.resize(nt);
      shadow.resize(nt);
    }
    vec_th[omp_get_thread_num()].resize(len);
  }

  int **restrict shadows_ptr=shadow.data();
  for(size_t tid=0; tid<shadow.size(); tid++)
  {
    int *restrict vec_ptr=vec_th[tid].data();
    // the explicit mapping is a workaround for a compiler bug
    #pragma omp target map(to:tid)
    {
      shadows_ptr[tid]=vec_ptr;
    }
  }

  const size_t nt=shadow.size();
  std::cout << "shadow size = " << nt << std::endl;
  // the explicit mapping is a workaround for a compiler bug
  #pragma omp target teams distribute map(to:nt)
  for(size_t iw=0; iw<nt; iw++)
  {
    #pragma omp parallel for
    for(size_t iel=0; iel<len; iel++)
      shadows_ptr[iw][iel] = iel+iw;
  }

  for(size_t tid=0; tid<shadow.size(); tid++)
  {
    vec_th[tid].update_from_device();
    std::cout << "iw = " << tid << " : ";
    for(size_t iel=0; iel<len; iel++)
      std::cout << "  " << vec_th[tid][iel];
    std::cout << std::endl;
  }
}
