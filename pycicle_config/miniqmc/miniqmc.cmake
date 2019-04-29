#  Copyright (c) 2019 Peter Doak
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Github settings
set(PYCICLE_GITHUB_PROJECT_NAME  "PDoakORNL/miniqmc")
set(PYCICLE_GITHUB_ORGANISATION  "PDoakORNL")
set(PYCICLE_GITHUB_BASE_BRANCH   "one_code")

# CDash server settings
set(PYCICLE_CDASH_PROJECT_NAME   "miniqmc")
set(PYCICLE_CDASH_SERVER_NAME    "cdash-minimal.ornl.gov")
set(PYCICLE_CDASH_HTTP_PATH      "cdash")

# project specific target to build before running tests
set(PYCICLE_CTEST_BUILD_TARGET   "all")
