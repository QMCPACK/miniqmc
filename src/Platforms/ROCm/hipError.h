//////////////////////////////////////////////////////////////////////////////////////
//// This file is distributed under the University of Illinois/NCSA Open Source License.
//// See LICENSE file in top directory for details.
////
//// Copyright (c) 2019 QMCPACK developers.
////
//// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
////
//// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_HIP_ERROR_H
#define QMCPLUSPLUS_HIP_ERROR_H

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <hip/hip_runtime_api.h>

#define hipErrorCheck(ans, cause)                \
  {                                               \
    hipAssert((ans), cause, __FILE__, __LINE__); \
  }
/// prints HIP error messages. Always use hipErrorCheck macro.
inline void hipAssert(hipError_t code, const std::string& cause, const char* filename, int line, bool abort = true)
{
  if (code != hipSuccess)
  {
    std::ostringstream err;
    err << "hipAssert: " << hipGetErrorName(code) << " " << hipGetErrorString(code) << ", file " << filename
        << ", line " << line << std::endl
        << cause << std::endl;
    std::cerr << err.str();
    if (abort)
      throw std::runtime_error(cause);
  }
}

#endif
