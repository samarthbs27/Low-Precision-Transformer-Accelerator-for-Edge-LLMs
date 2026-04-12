#ifndef TINYLLAMA_HLS_COMMON_STREAM_UTILS_HPP_
#define TINYLLAMA_HLS_COMMON_STREAM_UTILS_HPP_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fixed_types.hpp"

namespace tinyllama {
namespace hls_common {

template <typename T>
inline T clamp_value(T value, T min_value, T max_value) {
  if (value < min_value) {
    return min_value;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

inline act_int8_t saturate_to_int8(std::int32_t value) {
  constexpr std::int32_t kMin = -127;
  constexpr std::int32_t kMax = 127;
  std::int32_t clamped = clamp_value<std::int32_t>(value, kMin, kMax);
  return static_cast<act_int8_t>(clamped);
}

inline prob_int8_t saturate_probability_byte(std::int32_t value) {
  constexpr std::int32_t kMin = 0;
  constexpr std::int32_t kMax = kProbScale;
  std::int32_t clamped = clamp_value<std::int32_t>(value, kMin, kMax);
  return static_cast<prob_int8_t>(clamped);
}

inline std::int32_t round_to_nearest_even_fixed(fixed_t value) {
  fixed_t abs_value = (value < static_cast<fixed_t>(0))
                    ? -value
                    : value;
  std::int32_t floor_value = static_cast<std::int32_t>(abs_value);
  fixed_t frac_value = abs_value - static_cast<fixed_t>(floor_value);
  fixed_t half_value = static_cast<fixed_t>(1) / static_cast<fixed_t>(2);
  std::int32_t rounded_mag = floor_value;

  if (frac_value > half_value) {
    rounded_mag = floor_value + 1;
  } else if (frac_value == half_value) {
    rounded_mag = (floor_value & 1) ? (floor_value + 1) : floor_value;
  }

  return (value < static_cast<fixed_t>(0)) ? -rounded_mag : rounded_mag;
}

inline act_int8_t requantize_int8(std::int32_t value, fixed_t scale) {
  fixed_t scaled = static_cast<fixed_t>(value) * scale;
  return saturate_to_int8(round_to_nearest_even_fixed(scaled));
}

inline prob_int8_t quantize_probability(prob_fixed_t value) {
  prob_fixed_t scaled = value * static_cast<prob_fixed_t>(kProbScale);
  return saturate_probability_byte(round_to_nearest_even_fixed(scaled));
}

template <typename T, int N>
inline void copy_vec(const vec_t<T, N>& src, vec_t<T, N>* dst) {
  for (int i = 0; i < N; ++i) {
    dst->data[i] = src.data[i];
  }
}

template <typename T>
inline void write_blocking(stream_t<T>& stream, const T& value) {
  stream.write(value);
}

template <typename T>
inline T read_blocking(stream_t<T>& stream) {
  return stream.read();
}

template <typename T>
inline bool stream_has_data(const stream_t<T>& stream) {
  return !stream.empty();
}

}  // namespace hls_common
}  // namespace tinyllama

#endif  // TINYLLAMA_HLS_COMMON_STREAM_UTILS_HPP_
