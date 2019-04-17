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


/** @file OutputManager.h
 * @brief Declaration of OutputManager class.
 */
#ifndef OUTPUTMANAGER_H
#define OUTPUTMANAGER_H

#include <iostream>
#include <iostream>
#include <iomanip>

#include <Utilities/InfoStream.h>

enum class Verbosity
{
  LOW,
  HIGH,
  DEBUG
};
enum class LogType
{
  SUMMARY,
  APP,
  ERROR,
  DEBUG
};

class ISSingle
{
private:
  ISSingle() : infoSummary(&std::cout), infoLog(&std::cout), infoError(&std::cerr), infoDebug(&std::cout) {}
  ~ISSingle() {}
  ISSingle(const ISSingle&) = delete;
  ISSingle& operator=(const ISSingle&) = delete;
  ISSingle(ISSingle&)                  = delete;
  ISSingle& operator=(ISSingle&) = delete;

public:
  static ISSingle& get()
  {
    static ISSingle instance;
    return instance;
  }
  InfoStream infoSummary;
  InfoStream infoLog;
  InfoStream infoError;
  InfoStream infoDebug;
};

class OutputManagerClass
{
public:
  using IS = ISSingle;
  Verbosity global_verbosity_level;

private:
  OutputManagerClass() { setVerbosity(Verbosity::LOW); }
  OutputManagerClass(const OutputManagerClass&) = delete;
  OutputManagerClass& operator=(const OutputManagerClass&) = delete;

public:
  static OutputManagerClass& get()
  {
    static OutputManagerClass instance;
    ;
    return instance;
  }

  void setVerbosity(Verbosity level);

  bool isActive(Verbosity level);

  bool isDebugActive() { return isActive(Verbosity::DEBUG); }

  bool isHighActive() { return isActive(Verbosity::HIGH); }

  std::ostream& getStream(LogType log)
  {
    switch (log)
    {
    case LogType::SUMMARY:
      return IS::get().infoSummary.getStream();
    case LogType::APP:
      return IS::get().infoLog.getStream();
    case LogType::ERROR:
      return IS::get().infoError.getStream();
    case LogType::DEBUG:
      return IS::get().infoDebug.getStream();
    }
    return IS::get().infoDebug.getStream();
  }

  /// Pause the summary and log streams
  void pause();

  /// Resume the summary and log streams
  void resume();

  /// Permanently shut off all streams
  void shutOff();
};

namespace qmcplusplus
{
inline std::ostream& app_summary() { return OutputManagerClass::get().getStream(LogType::SUMMARY); }

inline std::ostream& app_log() { return OutputManagerClass::get().getStream(LogType::APP); }

inline std::ostream& app_error()
{
  OutputManagerClass::get().getStream(LogType::ERROR) << "ERROR ";
  return OutputManagerClass::get().getStream(LogType::ERROR);
}

inline std::ostream& app_warning()
{
  OutputManagerClass::get().getStream(LogType::ERROR) << "WARNING ";
  return OutputManagerClass::get().getStream(LogType::ERROR);
}

inline std::ostream& app_debug_stream() { return OutputManagerClass::get().getStream(LogType::DEBUG); }

// From https://stackoverflow.com/questions/11826554/standard-no-op-output-stream
// If debugging is not active, this skips evaluation of the arguments
#define app_debug                                    \
  if (!OutputManagerClass::get().isDebugActive()) {} \
  else                                               \
    app_debug_stream

}; // namespace qmcplusplus

#endif
