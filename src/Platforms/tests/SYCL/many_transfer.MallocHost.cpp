#include "catch.hpp"

#include <SYCL/SYCLruntime.hpp>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <timer.h>

#define N 32768
#define SEG_SIZE 1024

TEST_CASE("many_transfer.MallocHost", "[CUDA]")
{
  auto& q = qmcplusplus::getSYCLDefaultDeviceDefaultQueue();

  std::vector<double*> segments(N, nullptr);
  std::vector<double*> segments_dev(N, nullptr);
  for(int i = 0; i<N; i++)
  {
    segments[i] = sycl::malloc_host<double>(SEG_SIZE, q);
    segments_dev[i] = sycl::malloc_device<double>(SEG_SIZE, q);
    if(!segments[i])
      throw std::runtime_error("sycl::malloc_host<double> returns nullptr");
    if(!segments_dev[i])
      throw std::runtime_error("sycl::malloc_device<double> returns nullptr");
  }

  {
    Timer local("many_transfer.MallocHost first run");
    for(int i = 0; i<N; i++)
      q.copy<double>(segments[i], segments_dev[i], SEG_SIZE);
  }
  q.wait();

  {
    Timer local("many_transfer.MallocHost second run");
    for(int i = 0; i<N; i++)
      q.copy<double>(segments[i], segments_dev[i], SEG_SIZE);
  }
  q.wait();

  std::cout << "Success" << std::endl;

  for(int i = 0; i<N; i++)
  {
	  sycl::free(segments[i], q);
	  sycl::free(segments_dev[i], q);
  }
}
