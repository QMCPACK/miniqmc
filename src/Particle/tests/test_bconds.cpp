
#define CATCH_CONFIG_MAIN
#include "Utilities/unit/catch.hpp"

#include "Utilities/Configuration.h"
#include "Particle/Lattice/ParticleBConds.h"

#include "Particle/ParticleSet.h"
#include "Input/nio.hpp"

namespace qmcplusplus
{
TEST_CASE("test bconds", "[particle]")
{
  // clang-format off
  Tensor<double, 3> cell = {1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0};
  // clang-format on
  CrystalLattice<double, 3> lat;
  lat.set(cell);


  DTD_BConds<double, 3, PPPG + SOA_OFFSET> bc(lat);


  const int N = 2;
  TinyVector<double, 3> pos;
  VectorSoAContainer<double, 3> all_pos;

  double temp_r[N];
  VectorSoAContainer<double, 3> temp_dr;
  temp_dr.resize(2);
  int flip_index = 2;

  all_pos.resize(2);
  all_pos(0) = TinyVector<double, 3>({0.9, 0.0, 0.2});
  all_pos(1) = TinyVector<double, 3>({0.2, 0.4, 0.8});

  pos = {0.1, 0.2, 0.3};

  bc.computeDistances(pos, all_pos, temp_r, temp_dr, 0, N, flip_index);

  REQUIRE(temp_r[0] == Approx(0.3));
  REQUIRE(temp_r[1] == Approx(0.5477225575));

  REQUIRE(temp_dr[0][0] == Approx(-0.2));
  REQUIRE(temp_dr[0][1] == Approx(-0.2));
  REQUIRE(temp_dr[0][2] == Approx(-0.1));

  REQUIRE(temp_dr[1][0] == Approx(0.1));
  REQUIRE(temp_dr[1][1] == Approx(0.2));
  REQUIRE(temp_dr[1][2] == Approx(0.5));
  //std::cout << "temp_dir 0 = " << temp_dr[0] << std::endl;
  //std::cout << "temp_dir 1 = " << temp_dr[1] << std::endl;
  //std::cout << "temp_r 0 = " << temp_r[0] << std::endl;
  //std::cout << "temp_r 1 = " << temp_r[1] << std::endl;
}

TEST_CASE("test bconds general", "[particle]")
{
  // clang-format off
  Tensor<double, 3> cell = { 7.8811, 7.8811, 0.0,
                            -7.8811, 7.8811, 0.0,
                             0.0,    0.0,   15.7622};
  // clang-format on

  CrystalLattice<double, 3> lat;
  lat.set(cell);

  DTD_BConds<double, 3, PPPG + SOA_OFFSET> bc(lat);

  const int N = 2;
  TinyVector<double, 3> pos;
  VectorSoAContainer<double, 3> all_pos;

  double temp_r[N];
  VectorSoAContainer<double, 3> temp_dr;
  temp_dr.resize(2);
  int flip_index = 2;

  all_pos.resize(2);
  all_pos(0) = TinyVector<double, 3>({0.9, 0.0, 0.2});
  all_pos(1) = TinyVector<double, 3>({0.2, 0.4, 0.8});

  pos = {7.1, 7.2, 0.3};

  bc.computeDistances(pos, all_pos, temp_r, temp_dr, 0, N, flip_index);
  //std::cout << "temp_dir 0 = " << temp_dr[0] << std::endl;
  //std::cout << "temp_dir 1 = " << temp_dr[1] << std::endl;
  //std::cout << "temp_r 0 = " << temp_r[0] << std::endl;
  //std::cout << "temp_r 1 = " << temp_r[1] << std::endl;
}
} // namespace qmcplusplus
