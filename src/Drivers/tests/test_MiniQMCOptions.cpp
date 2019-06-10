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

#include <string>
#include <vector>
#include "catch.hpp"
#include "Drivers/MiniQMCOptions.h"

/** @file
 *
 *  For now we omit testing help and version output
 */

namespace qmcplusplus
{

TEST_CASE("MiniQMCOptions Read", "[Application]") {
    int argc = 26;
    std::vector<std::string> opt_strs = {"dummy_progname",
					 "-a","256",
					 "-b",
					 "-c","8",
					 "-g","2 1 3",
					 "-j",
					 "-m","256",
					 "-n","10",
					 "-N","1",
					 "-r","0.75",
					 "-s","140",
					 "-t","fine",
					 "-v",
					 "-w","20",
					 "-x","1.7"
    };
    
    std::vector<char*> option_ptrs;
    option_ptrs.reserve(argc + 1);
    std::transform(begin(opt_strs), end(opt_strs),
		   std::back_inserter(option_ptrs),
		   [](std::string& s) { char * ptr = new char[s.length() + 1];
		       std::strcpy(ptr, s.c_str() );
		       return ptr;});
    option_ptrs.push_back(nullptr);
    MiniQMCOptions mq_opt = readOptions(argc, option_ptrs.data());
    REQUIRE(mq_opt.valid);
    std::for_each(begin(option_ptrs), end(option_ptrs),
		  [](char* c) { delete[] c; });
    REQUIRE(mq_opt.splines_per_block == 256);
    REQUIRE(mq_opt.useRef);
    REQUIRE(mq_opt.crowd_size == 8);
    REQUIRE(mq_opt.na == 2);
    REQUIRE(mq_opt.nb == 1);
    REQUIRE(mq_opt.nc == 3);
    REQUIRE(mq_opt.enableJ3);
    REQUIRE(mq_opt.nx == 256 * 37); // PD: yes this 37 is hard coded into the options
    REQUIRE(mq_opt.ny == 256 * 37); // I carried it over from miniqmc_sync_move.cpp
    REQUIRE(mq_opt.nz == 256 * 37);
    REQUIRE(mq_opt.nsteps == 10);
    REQUIRE(mq_opt.nsubsteps == 1);
    REQUIRE(mq_opt.accept == 0.75);
    REQUIRE(mq_opt.iseed == 140);
    REQUIRE(mq_opt.timer_level_name == "fine");
    REQUIRE(mq_opt.verbose);
    REQUIRE(mq_opt.nmovers == 20);
    REQUIRE(mq_opt.Rmax == 1.7);
    
    
}

}

