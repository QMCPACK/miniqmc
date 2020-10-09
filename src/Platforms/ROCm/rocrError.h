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


#ifndef QMCPLUSPLUS_ROCR_ERROR_H
#define QMCPLUSPLUS_ROCR_ERROR_H

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <hsa.h>

#define rocrErrorCheck(ans, cause)                \
  {                                               \
    rocrAssert((ans), cause, __FILE__, __LINE__); \
  }
/// prints ROCR error messages. Always use rocrErrorCheck macro.
inline void rocrAssert(hsa_status_t code, const std::string& cause, const char* filename, int line, bool abort = true)
{
  if (code != HSA_STATUS_SUCCESS)
  {
    const char *errStr = nullptr;
    hsa_status_t errStr_status = hsa_status_string(code, &errStr);
    std::ostringstream err;
    err << "rocrAssert: " << (errStr_status == HSA_STATUS_SUCCESS ? errStr : "failed to get error string") << ", file " << filename
        << ", line " << line << std::endl
        << cause << std::endl;
    std::cerr << err.str();
    if (abort)
      throw std::runtime_error(cause);
  }
}

#endif
