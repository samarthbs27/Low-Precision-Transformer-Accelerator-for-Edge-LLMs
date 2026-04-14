#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../common/fixed_types.hpp"
#include "../common/test_memh.hpp"
#include "rmsnorm_core_hls.hpp"

namespace {

using tinyllama::hls_common::fixed_t;
using tinyllama::hls_common::hidden_vec32_t;
using tinyllama::hls_common::read_scalar_memh_i32;
using tinyllama::hls_common::read_scalar_memh_u32;
using tinyllama::hls_common::stream_t;
using tinyllama::rmsnorm::rmsnorm_core_hls;

fixed_t q16_to_fixed(std::int32_t raw_q16) {
  return static_cast<fixed_t>(static_cast<float>(raw_q16) / 65536.0f);
}

std::int32_t fixed_to_q16(fixed_t value) {
  return static_cast<std::int32_t>(std::lrint(static_cast<float>(value) * 65536.0f));
}

void expect_case(const std::string& case_base, int tolerance_lsb) {
  const std::vector<std::uint32_t> meta = read_scalar_memh_u32(case_base + ".meta.memh");
  const std::vector<std::int32_t> x_chunks = read_scalar_memh_i32(case_base + ".core_x_chunks.memh");
  const std::vector<std::int32_t> gamma_chunks = read_scalar_memh_i32(case_base + ".core_gamma_chunks.memh");
  const std::vector<std::int32_t> y_expected = read_scalar_memh_i32(case_base + ".core_y_chunks.memh");

  const int row_count = static_cast<int>(meta.at(0));
  const int feature_count = static_cast<int>(meta.at(1));
  const int feature_chunks = feature_count / 32;
  const int act_chunk_count = row_count * feature_chunks;

  stream_t<hidden_vec32_t> act_stream;
  stream_t<hidden_vec32_t> gamma_stream;
  stream_t<hidden_vec32_t> out_stream;

  for (int chunk_idx = 0; chunk_idx < feature_chunks; ++chunk_idx) {
    hidden_vec32_t chunk{};
    for (int elem = 0; elem < 32; ++elem) {
      chunk.data[elem] = q16_to_fixed(gamma_chunks[(chunk_idx * 32) + elem]);
    }
    gamma_stream.write(chunk);
  }

  for (int chunk_idx = 0; chunk_idx < act_chunk_count; ++chunk_idx) {
    hidden_vec32_t chunk{};
    for (int elem = 0; elem < 32; ++elem) {
      chunk.data[elem] = q16_to_fixed(x_chunks[(chunk_idx * 32) + elem]);
    }
    act_stream.write(chunk);
  }

  rmsnorm_core_hls(
      act_stream,
      gamma_stream,
      out_stream,
      static_cast<std::uint16_t>(row_count),
      static_cast<std::uint16_t>(feature_count),
      static_cast<fixed_t>(1.0e-5f));

  for (int chunk_idx = 0; chunk_idx < act_chunk_count; ++chunk_idx) {
    if (out_stream.empty()) {
      std::cerr << "RMSNorm output underflow for " << case_base << "\n";
      std::exit(1);
    }
    hidden_vec32_t chunk = out_stream.read();
    for (int elem = 0; elem < 32; ++elem) {
      const std::int32_t expected = y_expected[(chunk_idx * 32) + elem];
      const std::int32_t actual = fixed_to_q16(chunk.data[elem]);
      if (std::abs(actual - expected) > tolerance_lsb) {
        std::cerr << "RMSNorm mismatch for " << case_base
                  << " chunk=" << chunk_idx
                  << " elem=" << elem
                  << " expected=" << expected
                  << " actual=" << actual << "\n";
        std::exit(1);
      }
    }
  }
}

}  // namespace

int main() {
  expect_case("sim/golden_traces/phase5/rtl/phase5_prefill_layer0_rmsnorm1", 64);
  expect_case("sim/golden_traces/phase5/rtl/phase5_decode_layer0_rmsnorm1", 64);
  std::cout << "PASS: tb_rmsnorm\n";
  return 0;
}
