#include <string>
#include <limits>
#include "catch.hpp"
#include "Devices.h"
#include "Devices_HANA.hpp"
#include "boost/hana.hpp"
#include "Drivers/ComputationGraph.hpp"

namespace qmcplusplus
{
    namespace hana = boost::hana;
    TEST_CASE("ComputationGraph Instantiation", "[Driver]") {
	ComputationGraph cg;
    }
    TEST_CASE("ComputationGraph Instantiation", "[Driver]") {
	ComputationGraph cg;
	// show that from a string we can acquire the compile time Device enumeration value of that device
	// Which will be the correct no type template parameter to use.
	constexpr auto index = index_of(device_names, hana::string_c<'C','P','U'>);
	cg[std::string("WaveFunction")][std::string("DeterminantUpdate")] = static_cast<Devices>((int)(hana::value<decltype(index)>()));
    }

} // namespace qmcpluplus
