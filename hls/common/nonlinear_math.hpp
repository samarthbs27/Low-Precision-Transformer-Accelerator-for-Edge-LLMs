#ifndef TINYLLAMA_HLS_COMMON_NONLINEAR_MATH_HPP_
#define TINYLLAMA_HLS_COMMON_NONLINEAR_MATH_HPP_

#include <cstdint>

#if !TINYLLAMA_HLS_NATIVE_TYPES
  #include <cmath>
#endif

#include "fixed_types.hpp"

#if TINYLLAMA_HLS_NATIVE_TYPES && __has_include(<hls_math.h>)
  #include <hls_math.h>
#endif

namespace tinyllama {
namespace hls_common {

inline fixed_t fixed_sqrt(fixed_t value) {
#if TINYLLAMA_HLS_NATIVE_TYPES
  return hls::sqrt(value);
#else
  return static_cast<fixed_t>(std::sqrt(static_cast<float>(value)));
#endif
}

inline fixed_t fixed_exp(fixed_t value) {
#if TINYLLAMA_HLS_NATIVE_TYPES
  return hls::exp(value);
#else
  return static_cast<fixed_t>(std::exp(static_cast<float>(value)));
#endif
}

inline fixed_t fixed_sigmoid(fixed_t value) {
  fixed_t one = static_cast<fixed_t>(1);
  return one / (one + fixed_exp(-value));
}

template <typename T>
inline T max_value(T lhs, T rhs) {
  return (lhs > rhs) ? lhs : rhs;
}

template <typename T>
inline T min_value(T lhs, T rhs) {
  return (lhs < rhs) ? lhs : rhs;
}

template <typename T, int N>
inline void zero_vec(vec_t<T, N>* dst) {
  for (int idx = 0; idx < N; ++idx) {
    dst->data[idx] = static_cast<T>(0);
  }
}

}  // namespace hls_common
}  // namespace tinyllama

#endif  // TINYLLAMA_HLS_COMMON_NONLINEAR_MATH_HPP_
