//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2017 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include <Utilities/OutputManager.h>

void OutputManagerClass::setVerbosity(Verbosity level)
{
  global_verbosity_level = level;
  if (isActive(Verbosity::DEBUG))
  {
    IS::get().infoSummary.resume();
    IS::get().infoLog.resume();
    IS::get().infoDebug.resume();
  }
  else if (isActive(Verbosity::HIGH))
  {
    IS::get().infoSummary.resume();
    IS::get().infoLog.resume();
    IS::get().infoDebug.pause();
  }
  else if (isActive(Verbosity::LOW))
  {
    IS::get().infoSummary.resume();
    IS::get().infoLog.pause();
    IS::get().infoDebug.pause();
  }
}

bool OutputManagerClass::isActive(Verbosity level) { return level <= global_verbosity_level; }

void OutputManagerClass::pause()
{
  IS::get().infoSummary.pause();
  IS::get().infoLog.pause();
}

void OutputManagerClass::resume()
{
  IS::get().infoSummary.resume();
  IS::get().infoLog.resume();
}

void OutputManagerClass::shutOff()
{
  IS::get().infoSummary.shutOff();
  IS::get().infoLog.shutOff();
  IS::get().infoError.shutOff();
  IS::get().infoDebug.shutOff();
}
