#ifndef MINIQMC_COMPUTATION_GRAPH_HPP
#define MINIQMC_COMPUTATION_GRAPH_HPP
#include<map>
#include<string>
#include "Devices.h"

namespace qmcplusplus
{
    
class ComputationGraph
{
private:
    std::map<std::string, std::map<std::string, Devices>> the_map;
public:
    ComputationGraph() { the_map["WaveFunction"]; }

    std::map<std::string, Devices>& operator[](const std::string& computation_stage) { return the_map[computation_stage]; }
};

}
#endif
