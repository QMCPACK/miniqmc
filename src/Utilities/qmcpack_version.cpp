////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2017 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by:
// Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////

#include <Utilities/qmcpack_version.h>
#include <Host/OutputManager.h>

#define STR_EXPAND(x) #x
#define STR(x) STR_EXPAND(x)

// Wrapper around the auto-generated Git repository revision
// information file (git-rev.h)
// If not building from a git repository, the git-rev.h file is empty
#include "git-rev.h"

#ifdef GIT_BRANCH_RAW
#define QMCPACK_GIT_BRANCH STR(GIT_BRANCH_RAW)
#define QMCPACK_GIT_HASH STR(GIT_HASH_RAW)
#define QMCPACK_GIT_COMMIT_LAST_CHANGED STR(GIT_COMMIT_LAST_CHANGED_RAW)
#define QMCPACK_GIT_COMMIT_SUBJECT GIT_COMMIT_SUBJECT_RAW
#endif

using qmcplusplus::app_summary;
using std::endl;

void print_version(bool verbose)
{
#ifdef QMCPACK_GIT_BRANCH
  app_summary() << "miniqmc git branch: " << QMCPACK_GIT_BRANCH << endl;
  app_summary() << "miniqmc git commit: " << QMCPACK_GIT_HASH << endl;

  if (verbose)
  {
    app_summary() << "miniqmc git commit date: " << QMCPACK_GIT_COMMIT_LAST_CHANGED << endl;
    app_summary() << "miniqmc git commit subject: " << QMCPACK_GIT_COMMIT_SUBJECT << endl;
  }

#else
  app_summary() << "miniqmc not built from git repository" << endl;
#endif
  app_summary() << endl;
}
