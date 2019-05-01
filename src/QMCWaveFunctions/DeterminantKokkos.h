#ifndef QMCPLUSPLUS_DIRACDETERMINANT_KOKKOS_H
#define QMCPLUSPLUS_DIRACDETERMINANT_KOKKOS_H
#include <Kokkos_Core.hpp>

namespace qmcplusplus
{

struct DiracDeterminantKokkos : public QMCTraits
{
  using MatType = Kokkos::View<ValueType**, Kokkos::LayoutRight>;
  using DoubleMatType = Kokkos::View<double**, Kokkos::LayoutRight>;

  Kokkos::View<ValueType[1]> LogValue;
  Kokkos::View<ValueType[1]> curRatio;
  Kokkos::View<int[1]> FirstIndex;

  // inverse matrix to be updated
  MatType psiMinv;
  // storage for the row update 
  Kokkos::View<ValueType*> psiV;
  // temporary storage for row update
  Kokkos::View<ValueType*> tempRowVec;
  Kokkos::View<ValueType*> rcopy;
  // internal storage to perform inversion correctly
  MatType psiM;
  // temporary workspace for inversion
  MatType psiMsave;
  // temporary workspace for getrf
  Kokkos::View<ValueType*> getRfWorkSpace;
  Kokkos::View<ValueType**> getRiWorkSpace;
  // pivot array
  Kokkos::View<int*> piv;
  
  KOKKOS_INLINE_FUNCTION
  DiracDeterminantKokkos() { ; }

  KOKKOS_INLINE_FUNCTION
  DiracDeterminantKokkos& operator=(const DiracDeterminantKokkos& rhs) = default;

  DiracDeterminantKokkos(const DiracDeterminantKokkos&) = default;
};

};
#endif
