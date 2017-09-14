////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// ////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file einspline_spo.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_HPP
#include <Configuration.h>
#include <Particle/ParticleSet.h>
#include <spline2/bspline_allocator.hpp>
#include <spline2/MultiBsplineRef.hpp>
//#include <spline2/MultiBsplineOffload.hpp>
#include <simd/allocator.hpp>
#include "OhmmsPETE/OhmmsArray.h"
#include "OMP_target_test/OMPTinyVector.h"
#include "OMP_target_test/OMPVector.h"
#include "OMP_target_test/OMPVectorSoaContainer.h"
#include <iostream>

namespace qmcplusplus
{
template <typename T, typename compute_engine_type = MultiBsplineRef<T> >
struct einspline_spo
{
  /// define the einsplie data object type
  using self_type = einspline_spo<T, compute_engine_type>;
  using spline_type = typename bspline_traits<T, 3>::SplineType;
  using pos_type        = TinyVector<T, 3>;
  using vContainer_type = OMPVector<T, aligned_vector<T> >;
  using gContainer_type = OMPVectorSoaContainer<T, 3>;
  using hContainer_type = OMPVectorSoaContainer<T, 6>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// number of blocks
  int nBlocks;
  /// first logical block index
  int firstBlock;
  /// last gical block index
  int lastBlock;
  /// number of splines
  int nSplines;
  /// number of splines per block
  int nSplinesPerBlock;
  /// if true, responsible for cleaning up einsplines
  bool Owner;
  lattice_type Lattice;
  /// use allocator
  einspline::Allocator myAllocator;
  /// compute engine
  compute_engine_type compute_engine;

  OMPVector<spline_type *> einsplines;
  std::vector<vContainer_type> psi;
  std::vector<gContainer_type> grad;
  std::vector<hContainer_type> hess;

  // for shadows
  OMPVector<T*> psi_shadows;
  OMPVector<T*> grad_shadows;
  OMPVector<T*> hess_shadows;
  OMPVector<OMPTinyVector<T, 3> > u_shadows;

  /// default constructor
  einspline_spo()
      : nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false)
  {
  }
  /// disable copy constructor
  einspline_spo(const einspline_spo &in) = delete;
  /// disable copy operator
  einspline_spo &operator=(const einspline_spo &in) = delete;

  /** copy constructor
   * @param in einspline_spo
   * @param ncrews number of crews of a team
   * @param crewID id of this crew in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo(einspline_spo &in, int ncrews, int crewID)
      : Owner(false), Lattice(in.Lattice)
  {
    nSplines         = in.nSplines;
    nSplinesPerBlock = in.nSplinesPerBlock;
    nBlocks          = (in.nBlocks + ncrews - 1) / ncrews;
    firstBlock       = nBlocks * crewID;
    lastBlock        = std::min(in.nBlocks, nBlocks * (crewID + 1));
    nBlocks          = lastBlock - firstBlock;
    einsplines.resize(nBlocks);
    spline_type **restrict einsplines_ptr=einsplines.data();
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
    {
      einsplines[i] = in.einsplines[t];
      spline_type *restrict &tile_ptr=in.einsplines[t];
      #pragma omp target map(to:i) device(0)
      {
        einsplines_ptr[i]=tile_ptr;
      }
    }
    resize();
  }

  /// destructors
  ~einspline_spo()
  {
    if (Owner)
      for (int i = 0; i < nBlocks; ++i)
        myAllocator.destroy(einsplines[i]);
  }

  /// resize the containers
  void resize()
  {
    psi.resize(nBlocks);
    grad.resize(nBlocks);
    hess.resize(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
      psi[i].resize(nSplinesPerBlock);
      grad[i].resize(nSplinesPerBlock);
      hess[i].resize(nSplinesPerBlock);
    }
  }

  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int nblocks,
           bool init_random = true)
  {
    nSplines         = num_splines;
    nBlocks          = nblocks;
    nSplinesPerBlock = num_splines / nblocks;
    firstBlock       = 0;
    lastBlock        = nBlocks;
    if (einsplines.empty())
    {
      Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      pos_type start(0);
      pos_type end(1);
      einsplines.resize(nBlocks);
      spline_type **restrict einsplines_ptr=einsplines.data();
      RandomGenerator<T> myrandom(11);
      Array<T, 3> data(nx, ny, nz);
      std::fill(data.begin(), data.end(), T());
      myrandom.generate_uniform(data.data(), data.size());
      for (int i = 0; i < nBlocks; ++i)
      {
        einsplines[i] = myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random)
          for (int j = 0; j < nSplinesPerBlock; ++j)
            myAllocator.set(data.data(), einsplines[i], j);
        // attach pointers
        spline_type *restrict &tile_ptr=einsplines[i];
        T *restrict &coefs_ptr=einsplines[i]->coefs;
        #pragma omp target enter data map(to:tile_ptr[0:1],coefs_ptr[0:einsplines[i]->coefs_size]) device(0)
        //std::cout << "YYYY offload size = " << einsplines[i]->coefs_size << std::endl;
        #pragma omp target map(to:i) device(0)
        {
          einsplines_ptr[i]=tile_ptr;
          einsplines_ptr[i]->coefs=coefs_ptr;
        }
        //std::cout << "YYYY end of spline offloading" << std::endl;
      }
    }
    resize();
  }

  /** evaluate psi */
  inline void evaluate_v(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi */
  inline void evaluate_v_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_multi_vgh(const std::vector<pos_type> &p, std::vector<self_type *> &shadows)
  {
    const size_t nw = p.size();
    if(nw*nBlocks!=psi_shadows.size())
    {
      psi_shadows.resize(nw*nBlocks);
      grad_shadows.resize(nw*nBlocks);
      hess_shadows.resize(nw*nBlocks);

      T ** restrict psi_shadows_ptr=psi_shadows.data();
      T ** restrict grad_shadows_ptr=grad_shadows.data();
      T ** restrict hess_shadows_ptr=hess_shadows.data();
      for(size_t iw = 0; iw < nw; iw++)
      {
        auto &shadow = *shadows[iw];
        for (int i = 0; i < nBlocks; ++i)
        {
          T *restrict psi_ptr=shadow.psi[i].data();
          T *restrict grad_ptr=shadow.grad[i].data();
          T *restrict hess_ptr=shadow.hess[i].data();
          //std::cout << "psi_shadows_ptr mapped already? " << omp_target_is_present(psi_shadows_ptr,0) << std::endl;
          //std::cout << "psi_ptr mapped already? " << omp_target_is_present(psi_ptr,0) << std::endl;
          const size_t idx = iw*nBlocks+i;
          #pragma omp target map(to:idx) device(0)
          {
            psi_shadows_ptr[idx] = psi_ptr;
            grad_shadows_ptr[idx] = grad_ptr;
            hess_shadows_ptr[idx] = hess_ptr;
          }
        }
      }
    }

    u_shadows.resize(nw);
    for(size_t iw = 0; iw < nw; iw++)
      u_shadows[iw] = Lattice.toUnit(p[iw]);
    //std::cout << "mapped already? " << omp_target_is_present(u_shadows.data(),0) << std::endl;
    u_shadows.update_to_device();

    T ** restrict psi_shadows_ptr=psi_shadows.data();
    T ** restrict grad_shadows_ptr=grad_shadows.data();
    T ** restrict hess_shadows_ptr=hess_shadows.data();
    spline_type **einsplines_ptr=einsplines.data();
    OMPTinyVector<T, 3> *u_shadows_ptr=u_shadows.data();

    #pragma omp target teams distribute collapse(2) map(to:nw,nBlocks,nSplinesPerBlock) device(0)
    for(size_t iw = 0; iw < nw; iw++)
      for (int i = 0; i < nBlocks; ++i)
      {
        //#pragma omp parallel
        /*
        MultiBsplineOffload<T>::evaluate_vgh(einsplines_ptr[i],
                                             u_shadows_ptr[iw][0],
                                             u_shadows_ptr[iw][1],
                                             u_shadows_ptr[iw][2],
                                             psi_shadows_ptr[iw*nBlocks+i],
                                             grad_shadows_ptr[iw*nBlocks+i],
                                             hess_shadows_ptr[iw*nBlocks+i],
                                             nSplinesPerBlock);
        */
        const spline_type *restrict spline_m = einsplines_ptr[i];
        T                                  x = u_shadows_ptr[iw][0];
        T                                  y = u_shadows_ptr[iw][1];
        T                                  z = u_shadows_ptr[iw][2];
        T *restrict                     vals = psi_shadows_ptr[iw*nBlocks+i];
        T *restrict                    grads = grad_shadows_ptr[iw*nBlocks+i];
        T *restrict                     hess = hess_shadows_ptr[iw*nBlocks+i];
        size_t                   num_splines = nSplinesPerBlock;
        {

          int ix, iy, iz;
          T tx, ty, tz;
          T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];

          x -= spline_m->x_grid.start;
          y -= spline_m->y_grid.start;
          z -= spline_m->z_grid.start;
          SplineBound<T>::get(x * spline_m->x_grid.delta_inv, tx, ix,
                              spline_m->x_grid.num - 1);
          SplineBound<T>::get(y * spline_m->y_grid.delta_inv, ty, iy,
                              spline_m->y_grid.num - 1);
          SplineBound<T>::get(z * spline_m->z_grid.delta_inv, tz, iz,
                              spline_m->z_grid.num - 1);

          MultiBsplineData<T>::compute_prefactors(a, da, d2a, tx);
          MultiBsplineData<T>::compute_prefactors(b, db, d2b, ty);
          MultiBsplineData<T>::compute_prefactors(c, dc, d2c, tz);

          const intptr_t xs = spline_m->x_stride;
          const intptr_t ys = spline_m->y_stride;
          const intptr_t zs = spline_m->z_stride;

          const size_t out_offset = spline_m->num_splines;

          T *restrict gx = grads;
          T *restrict gy = grads + out_offset;
          T *restrict gz = grads + 2 * out_offset;

          T *restrict hxx = hess;
          T *restrict hxy = hess + out_offset;
          T *restrict hxz = hess + 2 * out_offset;
          T *restrict hyy = hess + 3 * out_offset;
          T *restrict hyz = hess + 4 * out_offset;
          T *restrict hzz = hess + 5 * out_offset;

          OMPstd::fill_n(vals,  out_offset  , T());
          OMPstd::fill_n(grads, out_offset*3, T());
          OMPstd::fill_n(hess,  out_offset*6, T());

          for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
            {
              const T *restrict coefs =
                  spline_m->coefs + (ix + i) * xs + (iy + j) * ys + iz * zs;
              const T *restrict coefszs  = coefs + zs;
              const T *restrict coefs2zs = coefs + 2 * zs;
              const T *restrict coefs3zs = coefs + 3 * zs;

              const T pre20 = d2a[i] * b[j];
              const T pre10 = da[i] * b[j];
              const T pre00 = a[i] * b[j];
              const T pre11 = da[i] * db[j];
              const T pre01 = a[i] * db[j];
              const T pre02 = a[i] * d2b[j];

              const int iSplitPoint = num_splines;
              for (int n = 0; n < iSplitPoint; n++)
              {

                T coefsv    = coefs[n];
                T coefsvzs  = coefszs[n];
                T coefsv2zs = coefs2zs[n];
                T coefsv3zs = coefs3zs[n];

                T sum0 = c[0] * coefsv + c[1] * coefsvzs + c[2] * coefsv2zs +
                         c[3] * coefsv3zs;
                T sum1 = dc[0] * coefsv + dc[1] * coefsvzs + dc[2] * coefsv2zs +
                         dc[3] * coefsv3zs;
                T sum2 = d2c[0] * coefsv + d2c[1] * coefsvzs + d2c[2] * coefsv2zs +
                         d2c[3] * coefsv3zs;

                hxx[n] += pre20 * sum0;
                hxy[n] += pre11 * sum0;
                hxz[n] += pre10 * sum1;
                hyy[n] += pre02 * sum0;
                hyz[n] += pre01 * sum1;
                hzz[n] += pre00 * sum2;
                gx[n] += pre10 * sum0;
                gy[n] += pre01 * sum0;
                gz[n] += pre00 * sum1;
                vals[n] += pre00 * sum0;
              }
            }

          const T dxInv = spline_m->x_grid.delta_inv;
          const T dyInv = spline_m->y_grid.delta_inv;
          const T dzInv = spline_m->z_grid.delta_inv;
          const T dxx   = dxInv * dxInv;
          const T dyy   = dyInv * dyInv;
          const T dzz   = dzInv * dzInv;
          const T dxy   = dxInv * dyInv;
          const T dxz   = dxInv * dzInv;
          const T dyz   = dyInv * dzInv;

          for (int n = 0; n < num_splines; n++)
          {
            gx[n] *= dxInv;
            gy[n] *= dyInv;
            gz[n] *= dzInv;
            hxx[n] *= dxx;
            hyy[n] *= dyy;
            hzz[n] *= dzz;
            hxy[n] *= dxy;
            hxz[n] *= dxz;
            hyz[n] *= dyz;
          }
        }
      }
    for(size_t iw = 0; iw < nw; iw++)
    {
      auto &shadow = *shadows[iw];
      for (int i = 0; i < nBlocks; ++i)
      {
        shadow.psi[i].update_from_device();
        shadow.grad[i].update_from_device();
        shadow.hess[i].update_from_device();
      }
    }
  }

  void print(std::ostream &os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock
       << " lastBlock=" << lastBlock << " nSplines=" << nSplines
       << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};
}

#endif
