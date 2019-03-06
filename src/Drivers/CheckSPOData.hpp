////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_CHECK_SPO_DATA_HPP
#define QMCPLUSPLUS_CHECK_SPO_DATA_HPP

namespace qmcplusplus
{

template<typename T>
struct CheckSPOData
{
  T ratio;
  T nspheremoves;
  T dNumVGHCalls;
  T evalV_v_err;
  T evalVGH_v_err;
  T evalVGH_g_err;
  T evalVGH_h_err;

  volatile CheckSPOData& operator+=(const volatile CheckSPOData& data) volatile
  {
    ratio += data.ratio;
    nspheremoves += data.nspheremoves;
    dNumVGHCalls += data.dNumVGHCalls;
    evalV_v_err += data.evalV_v_err;
    evalVGH_v_err += data.evalVGH_v_err;
    evalVGH_g_err += data.evalVGH_g_err;
    evalVGH_h_err += data.evalVGH_h_err;
    return *this;
  }

};
}
#endif
