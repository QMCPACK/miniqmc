////////////////////////////////////////////////////////////////////////////////
//// This file is distributed under the University of Illinois/NCSA Open Source
//// License.  See LICENSE file in top directory for details.
////
//// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
////
//// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory.
////
//// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory.
//////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_OMP_VECTOR_HPP
#define QMCPLUSPLUS_OMP_VECTOR_HPP

#include <vector>

namespace qmcplusplus
{

template<typename T, class Container = std::vector<T>>
class OMPVector:public Container
{
  private:
  size_t device_id;
  T * vec_ptr;

  public:
  inline OMPVector(size_t size = 0, size_t id = 0): device_id(id), vec_ptr(nullptr)
  {
    resize(size);
  }

  inline void resize(size_t size)
  {
    if(size!=Container::size())
    {
      if(Container::size()!=0)
      {
        #pragma omp target exit data map(delete:vec_ptr) device(device_id)
        vec_ptr = nullptr;
      }
      Container::resize(size);
      if(size>0)
      {
        vec_ptr = Container::data();
        #pragma omp target enter data map(alloc:vec_ptr[0:size]) device(device_id)
      }
    }
  }

  inline void update_to_device() const
  {
    #pragma omp target update to(vec_ptr[0:Container::size()]) device(device_id)
  }

  inline void update_from_device() const 
  {
    #pragma omp target update from(vec_ptr[0:Container::size()]) device(device_id)
  }

  inline ~OMPVector()
  {
    #pragma omp target exit data map(delete:vec_ptr) device(device_id)
  }

};

}

namespace OMPstd
{
#pragma omp declare target
  template <typename T>
  inline void fill_n(T *x, size_t count, const T& value)
  {
    for(size_t id=0; id<count; id++)
      x[id]=value;
  }
#pragma omp end declare target
}
#endif
