////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2017 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Mark Dewing, mdewing@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Mark Dewing, mdewing@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////

#include <Utilities/qmcpack_version.h>
#include <iostream>

using std::cout;
using std::endl;

void print_version(bool verbose)
{
#ifdef QMCPACK_GIT_BRANCH
  cout << "miniqmc git branch: " << QMCPACK_GIT_BRANCH << endl;
  cout << "miniqmc git commit: " << QMCPACK_GIT_HASH << endl;
  if (verbose) {
    cout << "miniqmc git commit date: " << QMCPACK_GIT_COMMIT_LAST_CHANGED << endl;
    cout << "miniqmc git commit subject: " << QMCPACK_GIT_COMMIT_SUBJECT << endl;
  }

#else
  cout << "miniqmc not built from git repository" << endl;
#endif
}
