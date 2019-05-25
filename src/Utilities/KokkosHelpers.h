#ifndef QMCPLUSPLUS_KOKKOS_HELPERS_H
#define QMCPLUSPLUS_KOKKOS_HELPERS_H


template<typename valType, int sz>
class structReductionHelper {
public:
  typedef structReductionHelper<valType,sz> value_type;
 private:
  valType data[sz];
public:
  KOKKOS_INLINE_FUNCTION
  structReductionHelper() {
    for (int i = 0; i < sz; i++) {
      data[i] = valType(0.0);
    }
  } 
  template<typename t2>
  KOKKOS_INLINE_FUNCTION
  structReductionHelper(const t2& val) {
    for (int i = 0; i < sz; i++) {
      data[i] = val;
    }
  }

  KOKKOS_INLINE_FUNCTION
  valType operator()(int i) const volatile { return data[i]; }
  KOKKOS_INLINE_FUNCTION
  volatile valType& operator()(int i) { return data[i]; }

  template<typename t2>
  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile structReductionHelper<t2,sz>& rhs) volatile {
    for (int i = 0; i < sz; i++) {
      data[i] += rhs(i);
    }
  }
  template<typename t2>
  KOKKOS_INLINE_FUNCTION
  structReductionHelper<valType,sz>& operator=(const structReductionHelper<t2,sz>& rhs) {
    for (int i = 0; i < sz; i++) {
      data[i] = rhs(i);
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  void init() const {
    for (int i = 0; i < sz; i++) {
      data[i] = valType(0.0);
    }
  }
 

};


template<typename T, int sz>
struct Kokkos::reduction_identity<structReductionHelper<T,sz>> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static structReductionHelper<T,sz> sum() { return structReductionHelper<T,sz>(T(0)); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static structReductionHelper<T,sz> prod() { return structReductionHelper<T,sz>(T(1)); }
};
 




#endif
