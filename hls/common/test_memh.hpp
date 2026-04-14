#ifndef TINYLLAMA_HLS_COMMON_TEST_MEMH_HPP_
#define TINYLLAMA_HLS_COMMON_TEST_MEMH_HPP_

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinyllama {
namespace hls_common {

inline std::vector<std::string> read_memh_lines(const std::string& path) {
  std::ifstream handle(path);
  if (!handle) {
    throw std::runtime_error("Failed to open memh file: " + path);
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(handle, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

inline std::int64_t parse_hex_signed(const std::string& token, int bits) {
  std::uint64_t value = 0;
  std::stringstream stream;
  stream << std::hex << token;
  stream >> value;

  if (bits < 64 && (value & (std::uint64_t(1) << (bits - 1)))) {
    const std::uint64_t mask = (std::uint64_t(1) << bits) - 1;
    value |= ~mask;
  }
  return static_cast<std::int64_t>(value);
}

inline std::uint64_t parse_hex_unsigned(const std::string& token) {
  std::uint64_t value = 0;
  std::stringstream stream;
  stream << std::hex << token;
  stream >> value;
  return value;
}

inline std::vector<std::int32_t> read_scalar_memh_i32(const std::string& path) {
  std::vector<std::int32_t> values;
  for (const std::string& line : read_memh_lines(path)) {
    values.push_back(static_cast<std::int32_t>(parse_hex_signed(line, 32)));
  }
  return values;
}

inline std::vector<std::uint32_t> read_scalar_memh_u32(const std::string& path) {
  std::vector<std::uint32_t> values;
  for (const std::string& line : read_memh_lines(path)) {
    values.push_back(static_cast<std::uint32_t>(parse_hex_unsigned(line)));
  }
  return values;
}

inline std::vector<std::int8_t> read_scalar_memh_i8(const std::string& path) {
  std::vector<std::int8_t> values;
  for (const std::string& line : read_memh_lines(path)) {
    values.push_back(static_cast<std::int8_t>(parse_hex_signed(line, 8)));
  }
  return values;
}

}  // namespace hls_common
}  // namespace tinyllama

#endif  // TINYLLAMA_HLS_COMMON_TEST_MEMH_HPP_
