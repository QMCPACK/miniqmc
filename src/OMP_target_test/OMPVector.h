
#ifndef QMCPLUSPLUS_OMPVECTOR_HPP
#define QMCPLUSPLUS_OMPVECTOR_HPP

#include <vector>

namespace qmcplusplus
{

template<typename T>
class OMPVector
{
  private:
  size_t device_id;
  T * vec_ptr;

  public:
  std::vector<T> vec;
  inline OMPVector(size_t size = 0, size_t id = 0): device_id(id), vec_ptr(nullptr)
  {
    resize(size);
  }

  inline void resize(size_t size)
  {
    if ( size > 0 )
    {
      vec.resize(size);
      vec_ptr = vec.data();
      #pragma omp target enter data map(alloc:vec_ptr[0:size]) device(device_id)
    }
  }

  inline void update_to_device() const
  {
    #pragma omp target update to(vec_ptr[0:vec.size()]) device(device_id)
  }

  inline void update_from_device() const 
  {
    #pragma omp target update from(vec_ptr[0:vec.size()]) device(device_id)
  }

  inline T * data() const
  {
    return vec_ptr;
  }

  inline ~OMPVector()
  {
    #pragma omp target exit data map(delete:vec_ptr) device(device_id)
  }
};

}
#endif
