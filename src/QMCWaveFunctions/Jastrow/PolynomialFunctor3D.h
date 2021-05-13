////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_POLYNOMIAL3D_FUNCTOR_H
#define QMCPLUSPLUS_POLYNOMIAL3D_FUNCTOR_H
#include "Numerics/OptimizableFunctorBase.h"
#include "Numerics/DeterminantOperators.h"
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include <cstdio>
#include <algorithm>

namespace qmcplusplus
{
struct PolynomialFunctor3D : public OptimizableFunctorBase
{
  typedef real_type value_type;
  int N_eI, N_ee;
  Array<real_type, 3> gamma;
  // Permutation vector, used when we need to pivot
  // columns
  std::vector<int> GammaPerm;

  Array<int, 3> index;
  std::vector<bool> IndepVar;
  std::vector<real_type> GammaVec, dval_Vec;
  std::vector<TinyVector<real_type, 3>> dgrad_Vec;
  std::vector<Tensor<real_type, 3>> dhess_Vec;
  int NumConstraints, NumGamma;
  Matrix<real_type> ConstraintMatrix;
  std::vector<real_type> Parameters, d_valsFD;
  std::vector<TinyVector<real_type, 3>> d_gradsFD;
  std::vector<Tensor<real_type, 3>> d_hessFD;
  std::vector<std::string> ParameterNames;
  std::string iSpecies, eSpecies1, eSpecies2;
  int ResetCount;
  real_type scale;
  // Order of continuity
  const int C;
  bool notOpt;

  /// constructor
  PolynomialFunctor3D(real_type ee_cusp = 0.0, real_type eI_cusp = 0.0)
      : N_eI(0), N_ee(0), ResetCount(0), scale(1.0), C(3), notOpt(false)
  {
    if (std::abs(ee_cusp) > 0.0 || std::abs(eI_cusp) > 0.0)
    {
      app_error() << "PolynomialFunctor3D does not support nonzero cusp.\n";
      abort();
    }
    cutoff_radius = 0.0;
  }

  void resize(int neI, int nee)
  {
    N_eI           = neI;
    N_ee           = nee;
    const double L = 0.5 * cutoff_radius;
    gamma.resize(N_eI + 1, N_eI + 1, N_ee + 1);
    index.resize(N_eI + 1, N_eI + 1, N_ee + 1);
    NumGamma       = ((N_eI + 1) * (N_eI + 2) / 2 * (N_ee + 1));
    NumConstraints = (2 * N_eI + 1) + (N_eI + N_ee + 1);
    int numParams  = NumGamma - NumConstraints;
    Parameters.resize(numParams);
    d_valsFD.resize(numParams);
    d_gradsFD.resize(numParams);
    d_hessFD.resize(numParams);
    GammaVec.resize(NumGamma);
    dval_Vec.resize(NumGamma);
    dgrad_Vec.resize(NumGamma);
    dhess_Vec.resize(NumGamma);
    ConstraintMatrix.resize(NumConstraints, NumGamma);
    // Assign indices
    int num = 0;
    for (int m = 0; m <= N_eI; m++)
      for (int l = m; l <= N_eI; l++)
        for (int n = 0; n <= N_ee; n++)
          index(l, m, n) = index(m, l, n) = num++;
    assert(num == NumGamma);
    //       std::cerr << "NumGamma = " << NumGamma << std::endl;
    // Fill up contraint matrix
    // For 3 constraints and 2 parameters, we would have
    // |A00 A01 A02 A03 A04|  |g0|   |0 |
    // |A11 A11 A12 A13 A14|  |g1|   |0 |
    // |A22 A21 A22 A23 A24|  |g2| = |0 |
    // | 0   0   0   1   0 |  |g3|   |p0|
    // | 0   0   0   0   1 |  |g4|   |p1|
    ConstraintMatrix = 0.0;
    int k;
    // e-e no-cusp constraint
    for (k = 0; k <= 2 * N_eI; k++)
    {
      for (int m = 0; m <= k; m++)
      {
        int l = k - m;
        if (l <= N_eI && m <= N_eI)
        {
          int i = index(l, m, 1);
          if (l > m)
            ConstraintMatrix(k, i) = 2.0;
          else if (l == m)
            ConstraintMatrix(k, i) = 1.0;
        }
      }
    }
    // e-I no-cusp constraint
    for (int kp = 0; kp <= N_eI + N_ee; kp++)
    {
      if (kp <= N_ee)
      {
        ConstraintMatrix(k + kp, index(0, 0, kp)) = (real_type)C;
        ConstraintMatrix(k + kp, index(0, 1, kp)) = -L;
      }
      for (int l = 1; l <= kp; l++)
      {
        int n = kp - l;
        if (n >= 0 && n <= N_ee && l <= N_eI)
        {
          ConstraintMatrix(k + kp, index(l, 0, n)) = (real_type)C;
          ConstraintMatrix(k + kp, index(l, 1, n)) = -L;
        }
      }
    }
    // Now, row-reduce constraint matrix
    GammaPerm.resize(NumGamma);
    IndepVar.resize(NumGamma, false);
    // Set identity permutation
    for (int i = 0; i < NumGamma; i++)
      GammaPerm[i] = i;
    int col = -1;
    for (int row = 0; row < NumConstraints; row++)
    {
      int max_loc;
      real_type max_abs;
      do
      {
        col++;
        max_loc = row;
        max_abs = std::abs(ConstraintMatrix(row, col));
        for (int ri = row + 1; ri < NumConstraints; ri++)
        {
          real_type abs_val = std::abs(ConstraintMatrix(ri, col));
          if (abs_val > max_abs)
          {
            max_loc = ri;
            max_abs = abs_val;
          }
        }
        if (max_abs < 1.0e-6)
          IndepVar[col] = true;
      } while (max_abs < 1.0e-6);
      ConstraintMatrix.swap_rows(row, max_loc);
      real_type lead_inv = 1.0 / ConstraintMatrix(row, col);
      for (int c = 0; c < NumGamma; c++)
        ConstraintMatrix(row, c) *= lead_inv;
      // Now, eliminate column entries
      for (int ri = 0; ri < NumConstraints; ri++)
      {
        if (ri != row)
        {
          real_type val = ConstraintMatrix(ri, col);
          for (int c = 0; c < NumGamma; c++)
            ConstraintMatrix(ri, c) -= val * ConstraintMatrix(row, c);
        }
      }
    }
    for (int c = col + 1; c < NumGamma; c++)
      IndepVar[c] = true;
  }

  void reset()
  {
    resize(N_eI, N_ee);
    reset_gamma();
  }

  void reset_gamma()
  {
    const double L = 0.5 * cutoff_radius;
    std::fill(GammaVec.begin(), GammaVec.end(), 0.0);
    // First, set all independent variables
    int var = 0;
    for (int i = 0; i < NumGamma; i++)
      if (IndepVar[i])
        GammaVec[i] = scale * Parameters[var++];
    assert(var == Parameters.size());
    // Now, set dependent variables
    var = 0;
    for (int i = 0; i < NumGamma; i++)
      if (!IndepVar[i])
      {
        assert(std::abs(ConstraintMatrix(var, i) - 1.0) < 1.0e-6);
        for (int j = 0; j < NumGamma; j++)
          if (i != j)
            GammaVec[i] -= ConstraintMatrix(var, j) * GammaVec[j];
        var++;
      }
    int num = 0;
    for (int m = 0; m <= N_eI; m++)
      for (int l = m; l <= N_eI; l++)
        for (int n = 0; n <= N_ee; n++)
          //	    gamma(m,l,n) = gamma(l,m,n) = unpermuted[num++];
          gamma(m, l, n) = gamma(l, m, n) = GammaVec[num++];
    // Now check that constraints have been satisfied
    // e-e constraints
    for (int k = 0; k <= 2 * N_eI; k++)
    {
      real_type sum = 0.0;
      for (int m = 0; m <= k; m++)
      {
        int l = k - m;
        if (l <= N_eI && m <= N_eI)
        {
          int i = index(l, m, 1);
          if (l > m)
            sum += 2.0 * GammaVec[i];
          else if (l == m)
            sum += GammaVec[i];
        }
      }
      if (std::abs(sum) > 1.0e-9)
        std::cerr << "error in k = " << k << "  sum = " << sum << std::endl;
    }
    for (int k = 0; k <= 2 * N_eI; k++)
    {
      real_type sum = 0.0;
      for (int l = 0; l <= k; l++)
      {
        int m = k - l;
        if (m <= N_eI && l <= N_eI)
        {
          sum += gamma(l, m, 1);
        }
      }
      if (std::abs(sum) > 1.0e-6)
      {
        app_error() << "e-e constraint not satisfied in PolynomialFunctor3D:  k=" << k
                    << "  sum=" << sum << std::endl;
        abort();
      }
    }
    // e-I constraints
    for (int k = 0; k <= N_eI + N_ee; k++)
    {
      real_type sum = 0.0;
      for (int m = 0; m <= k; m++)
      {
        int n = k - m;
        if (m <= N_eI && n <= N_ee)
        {
          sum += (real_type)C * gamma(0, m, n) - L * gamma(1, m, n);
        }
      }
      if (std::abs(sum) > 1.0e-6)
      {
        app_error() << "e-I constraint not satisfied in PolynomialFunctor3D:  k=" << k
                    << "  sum=" << sum << std::endl;
        abort();
      }
    }
  }

  inline real_type evaluate(real_type r_12, real_type r_1I, real_type r_2I) const
  {
    constexpr real_type czero(0);
    constexpr real_type cone(1);
    constexpr real_type chalf(0.5);

    const real_type L = chalf * cutoff_radius;
    if (r_1I >= L || r_2I >= L)
      return czero;
    real_type val = czero;
    real_type r2l(cone);
    for (int l = 0; l <= N_eI; l++)
    {
      real_type r2m(r2l);
      for (int m = 0; m <= N_eI; m++)
      {
        real_type r2n(r2m);
        for (int n = 0; n <= N_ee; n++)
        {
          val += gamma(l, m, n) * r2n;
          r2n *= r_12;
        }
        r2m *= r_2I;
      }
      r2l *= r_1I;
    }
    for (int i = 0; i < C; i++)
      val *= (r_1I - L) * (r_2I - L);
    return val;
  }

  // assume r_1I < L && r_2I < L, compression and screening is handled outside
  inline real_type evaluateV(int Nptcl,
                             const real_type* restrict r_12_array,
                             const real_type* restrict r_1I_array,
                             const real_type* restrict r_2I_array) const
  {
    constexpr real_type czero(0);
    constexpr real_type cone(1);
    constexpr real_type chalf(0.5);

    const real_type L = chalf * cutoff_radius;
    real_type val_tot = czero;

    //#pragma omp simd aligned(r_12_array,r_1I_array,r_2I_array) reduction(+:val_tot)
    for (int ptcl = 0; ptcl < Nptcl; ptcl++)
    {
      const real_type r_12 = r_12_array[ptcl];
      const real_type r_1I = r_1I_array[ptcl];
      const real_type r_2I = r_2I_array[ptcl];
      real_type val        = czero;
      real_type r2l(cone);
      for (int l = 0; l <= N_eI; l++)
      {
        real_type r2m(r2l);
        for (int m = 0; m <= N_eI; m++)
        {
          real_type r2n(r2m);
          for (int n = 0; n <= N_ee; n++)
          {
            val += gamma(l, m, n) * r2n;
            r2n *= r_12;
          }
          r2m *= r_2I;
        }
        r2l *= r_1I;
      }
      const real_type both_minus_L = (r_2I - L) * (r_1I - L);
      for (int i = 0; i < C; i++)
        val *= both_minus_L;
      val_tot += val;
    }

    return val_tot;
  }

  inline real_type evaluate(real_type r_12,
                            real_type r_1I,
                            real_type r_2I,
                            TinyVector<real_type, 3>& grad,
                            Tensor<real_type, 3>& hess) const
  {
    constexpr real_type czero(0);
    constexpr real_type cone(1);
    constexpr real_type chalf(0.5);
    constexpr real_type ctwo(2);

    const real_type L = chalf * cutoff_radius;
    if (r_1I >= L || r_2I >= L)
    {
      grad = czero;
      hess = czero;
      return czero;
    }
    real_type val = czero;
    grad          = czero;
    hess          = czero;
    real_type r2l(cone), r2l_1(czero), r2l_2(czero), lf(czero);
    for (int l = 0; l <= N_eI; l++)
    {
      real_type r2m(cone), r2m_1(czero), r2m_2(czero), mf(czero);
      for (int m = 0; m <= N_eI; m++)
      {
        real_type r2n(cone), r2n_1(czero), r2n_2(czero), nf(czero);
        for (int n = 0; n <= N_ee; n++)
        {
          const real_type g    = gamma(l, m, n);
          const real_type g00x = g * r2l * r2m;
          const real_type g10x = g * r2l_1 * r2m;
          const real_type g01x = g * r2l * r2m_1;
          const real_type gxx0 = g * r2n;

          val += g00x * r2n;
          grad[0] += g00x * r2n_1;
          grad[1] += g10x * r2n;
          grad[2] += g01x * r2n;
          hess(0, 0) += g00x * r2n_2;
          hess(0, 1) += g10x * r2n_1;
          hess(0, 2) += g01x * r2n_1;
          hess(1, 1) += gxx0 * r2l_2 * r2m;
          hess(1, 2) += gxx0 * r2l_1 * r2m_1;
          hess(2, 2) += gxx0 * r2l * r2m_2;
          nf += cone;
          r2n_2 = r2n_1 * nf;
          r2n_1 = r2n * nf;
          r2n *= r_12;
        }
        mf += cone;
        r2m_2 = r2m_1 * mf;
        r2m_1 = r2m * mf;
        r2m *= r_2I;
      }
      lf += cone;
      r2l_2 = r2l_1 * lf;
      r2l_1 = r2l * lf;
      r2l *= r_1I;
    }

    const real_type r_2I_minus_L = r_2I - L;
    const real_type r_1I_minus_L = r_1I - L;
    const real_type both_minus_L = r_2I_minus_L * r_1I_minus_L;
    for (int i = 0; i < C; i++)
    {
      hess(0, 0) = both_minus_L * hess(0, 0);
      hess(0, 1) = both_minus_L * hess(0, 1) + r_2I_minus_L * grad[0];
      hess(0, 2) = both_minus_L * hess(0, 2) + r_1I_minus_L * grad[0];
      hess(1, 1) = both_minus_L * hess(1, 1) + ctwo * r_2I_minus_L * grad[1];
      hess(1, 2) = both_minus_L * hess(1, 2) + r_1I_minus_L * grad[1] + r_2I_minus_L * grad[2] + val;
      hess(2, 2) = both_minus_L * hess(2, 2) + ctwo * r_1I_minus_L * grad[2];
      grad[0]    = both_minus_L * grad[0];
      grad[1]    = both_minus_L * grad[1] + r_2I_minus_L * val;
      grad[2]    = both_minus_L * grad[2] + r_1I_minus_L * val;
      val *= both_minus_L;
    }
    hess(1, 0) = hess(0, 1);
    hess(2, 0) = hess(0, 2);
    hess(2, 1) = hess(1, 2);
    return val;
  }

  // assume r_1I < L && r_2I < L, compression and screening is handled outside
  inline void evaluateVGL(int Nptcl,
                          const real_type* restrict r_12_array,
                          const real_type* restrict r_1I_array,
                          const real_type* restrict r_2I_array,
                          real_type* restrict val_array,
                          real_type* restrict grad0_array,
                          real_type* restrict grad1_array,
                          real_type* restrict grad2_array,
                          real_type* restrict hess00_array,
                          real_type* restrict hess11_array,
                          real_type* restrict hess22_array,
                          real_type* restrict hess01_array,
                          real_type* restrict hess02_array) const
  {
    constexpr real_type czero(0);
    constexpr real_type cone(1);
    constexpr real_type chalf(0.5);
    constexpr real_type ctwo(2);

    const real_type L = chalf * cutoff_radius;
    #pragma omp simd aligned(r_12_array,   \
                             r_1I_array,   \
                             r_2I_array,   \
                             val_array,    \
                             grad0_array,  \
                             grad1_array,  \
                             grad2_array,  \
                             hess00_array, \
                             hess11_array, \
                             hess22_array, \
                             hess01_array, \
                             hess02_array)
    for (int ptcl = 0; ptcl < Nptcl; ptcl++)
    {
      const real_type r_12 = r_12_array[ptcl];
      const real_type r_1I = r_1I_array[ptcl];
      const real_type r_2I = r_2I_array[ptcl];

      real_type val(czero);
      real_type grad0(czero);
      real_type grad1(czero);
      real_type grad2(czero);
      real_type hess00(czero);
      real_type hess11(czero);
      real_type hess22(czero);
      real_type hess01(czero);
      real_type hess02(czero);

      real_type r2l(cone), r2l_1(czero), r2l_2(czero), lf(czero);
      for (int l = 0; l <= N_eI; l++)
      {
        real_type r2m(cone), r2m_1(czero), r2m_2(czero), mf(czero);
        for (int m = 0; m <= N_eI; m++)
        {
          real_type r2n(cone), r2n_1(czero), r2n_2(czero), nf(czero);
          for (int n = 0; n <= N_ee; n++)
          {
            const real_type g    = gamma(l, m, n);
            const real_type g00x = g * r2l * r2m;
            const real_type g10x = g * r2l_1 * r2m;
            const real_type g01x = g * r2l * r2m_1;
            const real_type gxx0 = g * r2n;

            val += g00x * r2n;
            grad0 += g00x * r2n_1;
            grad1 += g10x * r2n;
            grad2 += g01x * r2n;
            hess00 += g00x * r2n_2;
            hess01 += g10x * r2n_1;
            hess02 += g01x * r2n_1;
            hess11 += gxx0 * r2l_2 * r2m;
            hess22 += gxx0 * r2l * r2m_2;
            nf += cone;
            r2n_2 = r2n_1 * nf;
            r2n_1 = r2n * nf;
            r2n *= r_12;
          }
          mf += cone;
          r2m_2 = r2m_1 * mf;
          r2m_1 = r2m * mf;
          r2m *= r_2I;
        }
        lf += cone;
        r2l_2 = r2l_1 * lf;
        r2l_1 = r2l * lf;
        r2l *= r_1I;
      }

      const real_type r_2I_minus_L = r_2I - L;
      const real_type r_1I_minus_L = r_1I - L;
      const real_type both_minus_L = r_2I_minus_L * r_1I_minus_L;
      for (int i = 0; i < C; i++)
      {
        hess00 = both_minus_L * hess00;
        hess01 = both_minus_L * hess01 + r_2I_minus_L * grad0;
        hess02 = both_minus_L * hess02 + r_1I_minus_L * grad0;
        hess11 = both_minus_L * hess11 + ctwo * r_2I_minus_L * grad1;
        hess22 = both_minus_L * hess22 + ctwo * r_1I_minus_L * grad2;
        grad0  = both_minus_L * grad0;
        grad1  = both_minus_L * grad1 + r_2I_minus_L * val;
        grad2  = both_minus_L * grad2 + r_1I_minus_L * val;
        val *= both_minus_L;
      }

      val_array[ptcl]    = val;
      grad0_array[ptcl]  = grad0 / r_12;
      grad1_array[ptcl]  = grad1 / r_1I;
      grad2_array[ptcl]  = grad2 / r_2I;
      hess00_array[ptcl] = hess00;
      hess11_array[ptcl] = hess11;
      hess22_array[ptcl] = hess22;
      hess01_array[ptcl] = hess01 / (r_12 * r_1I);
      hess02_array[ptcl] = hess02 / (r_12 * r_2I);
    }
  }
};
} // namespace qmcplusplus
#endif
