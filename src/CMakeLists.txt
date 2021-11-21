#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2019 QMCPACK developers.
#//
#// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
#//		       Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//                    Marl Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
#//                    Ying Wai Li, yingwaili@ornl.gov, Oak Ridge National Laboratory
#//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
#//                    Norbert Podhorszki, pnorbert@ornl.gov, Oak Ridge National Laboratory
#//                    Cynthia Gu, zg1@ornl.gov, Oak Ridge National Laboratory
#//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
#//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign  
#//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
#//
#// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
#//////////////////////////////////////////////////////////////////////////////////////

INCLUDE(${PROJECT_SOURCE_DIR}/CMake/unit_test.cmake)

CONFIGURE_FILE(${miniqmc_SOURCE_DIR}/src/config.h.cmake.in
  ${miniqmc_BINARY_DIR}/src/config.h)

SET(GITREV_BARE_FILE git-rev.h)
SET(GITREV_BARE_TMP git-rev-tmp.h)
SET(GITREV_FILE ${CMAKE_BINARY_DIR}/src/${GITREV_BARE_FILE})
SET(GITREV_TMP ${CMAKE_BINARY_DIR}/src/${GITREV_BARE_TMP})

# Always clean up git-rev-tmp.h file when cmake is run
EXECUTE_PROCESS(
  COMMAND ${CMAKE_COMMAND} -E remove ${GITREV_TMP}
)

IF (IS_GIT_PROJECT)
  # The following custom command picks up changes to the git revision information
  # every time the project is rebuilt. Even if the repositiory is updated (git pull)
  # without re-running cmake. It also appends '-dirty' to the commit hash if there are
  # unsaved changes to the repository.
  #
  # To avoid triggering a relink every time, the repository info is saved to
  # a temporary file, and the temporary file is copied over the actual file
  # only if the contents changed (using 'copy_if_different').
  # The temporary file is deleted to force the custom command to run on
  # the next build.
  #
  # Apparently custom commands need to be defined where the output is used.
  # If this in the main CMakeLists.txt it does not work.

  # Sed flags were once an issue and some HPC have old sed's
  set(SED_FLAG "-E")
  EXECUTE_PROCESS(COMMAND "sed" ${SED_FLAG} "s/\"/\\\\\"/g" "<${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    OUTPUT_QUIET
    ERROR_VARIABLE SED_ERROR)
  if ( SED_ERROR MATCHES ".*invalid.*" )
    set(SED_FLAG "-r")
    EXECUTE_PROCESS(COMMAND "sed" ${SED_FLAG} "s/\"/\\\\\"/g" "<${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
      OUTPUT_QUIET ERROR_QUIET
      ERROR_VARIABLE SED_ERROR)
    if ( SED_ERROR MATCHES ".*invalid.*" )
      MESSAGE(WARNING "Your system supports neither the sed -E or -r flag, git revision information will not be included in output")
    else ( SED_ERROR MATCHES ".*invalid.*" )
      MESSAGE("sed supports -r")
    endif ( SED_ERROR MATCHES ".*invalid.*" )
  else ( SED_ERROR MATCHES ".*invalid.*" )
    MESSAGE("sed supports -E")
  endif ( SED_ERROR MATCHES ".*invalid.*" )

  ADD_CUSTOM_COMMAND(
    OUTPUT ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo_append "#define GIT_BRANCH_RAW " > ${GITREV_TMP}
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo_append "#define GIT_HASH_RAW " >> ${GITREV_TMP}
    COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev=40 --match="NoTagWithThisName" >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo_append "#define GIT_COMMIT_LAST_CHANGED_RAW " >> ${GITREV_TMP}
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%ad >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo_append "#define GIT_COMMIT_SUBJECT_RAW \"" >> ${GITREV_TMP}
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%s | sed ${SED_FLAG} "s/\"/\\\\\"/g" | tr -d '\\n' >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo_append "\"" >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E echo >> ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GITREV_TMP} ${GITREV_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove ${GITREV_TMP}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
  )

  # Print some configure-time git info (useful for understanding what commits
  # are in particular build for the nightly CDash reports)

  EXECUTE_PROCESS(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    OUTPUT_VARIABLE GIT_CONFIG_BRANCH
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Breaking down the arguments to 'git describe'
  #  --abbrev=40     Size of hash to print.  This should print the entire hash.
  #  --dirty         Append hash with '-dirty' if there are uncommited changes.
  # The behavior of describe looks for a tag in the parents first, and then falls
  # back to the commit hash (if --always is specified)
  #  --always        Show the commit hash as fallback
  #  --match="NoTagWithThisName"
  #     If a tag is found, the output looks like:
  #       second_annotated_tag-29-g1fd38cccc0fd2f683ec223ca0783bb671bfedd4e
  #     In order to always get just the commit hash, specify a tag pattern
  #     that should never match.
  EXECUTE_PROCESS(
    COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev=40 --match="NoTagWithThisName"
    OUTPUT_VARIABLE GIT_CONFIG_COMMIT_HASH
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  MESSAGE("Git branch: ${GIT_CONFIG_BRANCH}")
  MESSAGE("Git commit hash: ${GIT_CONFIG_COMMIT_HASH}")
ELSE()
  # Output a blank git version file
  EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E remove ${GITREV_FILE}
    COMMAND ${CMAKE_COMMAND} -E touch ${GITREV_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E touch ${GITREV_TMP}
  )
ENDIF()


  ####################################
  # create libqmcutil
  ####################################
  SET(UTILITIES
    Utilities/SpeciesSet.cpp 
    Utilities/Communicate.cpp
    Utilities/NewTimer.cpp
    Utilities/XMLWriter.cpp
    Utilities/tinyxml/tinyxml2.cpp
    Utilities/qmcpack_version.cpp
    ${GITREV_TMP}
    )

  ADD_LIBRARY(qmcutil ${UTILITIES})
  TARGET_LINK_LIBRARIES(qmcutil PUBLIC platform_runtime
                                INTERFACE qmc_mpi_cxx)

  SUBDIRS(Utilities)

  # For unit tests, enable use for the fake RNG
  FUNCTION( USE_FAKE_RNG TARGET )
    TARGET_COMPILE_DEFINITIONS(${TARGET} PRIVATE "USE_FAKE_RNG")
  ENDFUNCTION()

  # Put the fake RNG in a separate library so production code doesn't
  # accidentally link to it

  #IF(QMC_BUILD_STATIC)
  #  ADD_LIBRARY(qmcutil STATIC ${UTILITIES})
  #ELSE(QMC_BUILD_STATIC)
  #  ADD_LIBRARY(qmcutil SHARED ${UTILITIES})
  #ENDIF(QMC_BUILD_STATIC)


  ####################################
  # create libqmcbase 
  ####################################
  SET(PARTICLE ${PARTICLE}
    Particle/VirtualParticleSet.cpp 
    Particle/ParticleSet.cpp 
    Particle/ParticleSet.BC.cpp 
    Particle/DistanceTableAA.cpp
    Particle/DistanceTableAB.cpp
    Particle/ParticleSet_builder.cpp
    )

  ADD_LIBRARY(qmcbase ${PARTICLE})
  TARGET_LINK_LIBRARIES(qmcbase PUBLIC qmcutil)

  SUBDIRS(Particle/tests)

  SUBDIRS(Platforms)
  SUBDIRS(QMCWaveFunctions)
  SUBDIRS(Drivers)
