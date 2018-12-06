#include "Drivers/Mover.hpp"

namespace qmcplusplus
{
template<class T, typename TBOOL>
const std::vector<T*>
    filtered_list(const std::vector<T*>& input_list, const std::vector<TBOOL>& chosen)
{
  std::vector<T*> final_list;
  for (int iw = 0; iw < input_list.size(); iw++)
    if (chosen[iw])
      final_list.push_back(input_list[iw]);
  return final_list;
}

const std::vector<ParticleSet*> extract_els_list(const std::vector<Mover*>& mover_list)
{
  std::vector<ParticleSet*> els_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    els_list.push_back(&(*it)->els);
  return els_list;
}

const std::vector<SPOSet*> extract_spo_list(const std::vector<Mover*>& mover_list)
{
  std::vector<SPOSet*> spo_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    spo_list.push_back((*it)->spo);
  return spo_list;
}

const std::vector<WaveFunction*> extract_wf_list(const std::vector<Mover*>& mover_list)
{
  std::vector<WaveFunction*> wf_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    wf_list.push_back(&(*it)->wavefunction);
  return wf_list;
}

} // namespace qmcplusplus
