#ifndef QMCPLUSPLUS_CHECKMULTIBSPLINEEVALOUTPUT_HPP
#define QMCPLUSPLUS_CHECKMULTIBSPLINEEVALOUTPUT_HPP

#include "Numerics/Containers.h"

namespace qmcplusplus
{
template<typename T>
struct CheckMultiBsplineEvalOutput
{
  bool operator()(const aligned_vector<T>& vals, const aligned_vector<T>& gpu_vals,
                  const VectorSoAContainer<T, 3>& grads, const VectorSoAContainer<T, 3>& gpu_grads,
                  const VectorSoAContainer<T, 6>& hess, const VectorSoAContainer<T, 6>& gpu_hess)
  {
    bool matching_vals = true;
    for (int i = 0; i < gpu_vals.size(); ++i)
    {
      //std::cout << vals[i] << " : " << gpu_vals[i] << '\n';
      if (vals[i] != Approx(gpu_vals[i]).epsilon(0.005))
      {
        bool matching_spline_vals = false;
        std::cout << "evaluation values do not match (cpu : gpu)  " << vals[i] << " : "
                  << gpu_vals[i] << '\n';
        break;
      }
    }
    REQUIRE(matching_vals);

    bool matching_grads = true;
    for (int i = 0; i < gpu_grads.size(); ++i)
    {
      if (matching_grads)
        for (int j = 0; j < 3; ++j)
        {
          if (matching_grads)
            if (grads[i][j] != Approx(gpu_grads[i][j]).epsilon(0.005))
            {
              matching_grads = false;
              std::cout << "eval_vgh grad ( " << i << "," << j << " ) does not match cpu : gpu "
                        << grads[i][j] << " : " << gpu_grads[i][j] << '\n';
              break;
            }
        }
    }

    REQUIRE(matching_grads);

    bool matching_hessian = true;
    for (int i = 0; i < gpu_hess.size(); ++i)
    {
      if (matching_hessian)
        for (int j = 0; j < 6; ++j)
        {
          if (matching_hessian)
            if (hess[i][j] != Approx(gpu_hess[i][j]).epsilon(0.01))
            {
              matching_hessian = false;
              std::cout << "eval_vgh hessian ( " << i << "," << j << " ) does not match cpu : gpu "
                        << std::setprecision(14) << hess[i][j] << " : " << gpu_hess[i][j] << '\n';

              break;
            }
        }
    }
    REQUIRE(matching_hessian);

    return true;
  }
};

} // namespace qmcplusplus
#endif
