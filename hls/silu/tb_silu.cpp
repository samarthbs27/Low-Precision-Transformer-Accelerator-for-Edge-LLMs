#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../common/fixed_types.hpp"
#include "../common/test_memh.hpp"
#include "silu_core_hls.hpp"

namespace {

using tinyllama::hls_common::fixed_t;
using tinyllama::hls_common::hidden_vec32_t;
using tinyllama::hls_common::read_scalar_memh_i32;
using tinyllama::hls_common::read_scalar_memh_u32;
using tinyllama::hls_common::stream_t;
using tinyllama::silu::silu_core_hls;

fixed_t q16_to_fixed(std::int32_t raw_q16) {
  return static_cast<fixed_t>(static_cast<float>(raw_q16) / 65536.0f);
}

std::int32_t fixed_to_q16(fixed_t value) {
  return static_cast<std::int32_t>(std::lrint(static_cast<float>(value) * 65536.0f));
}

void expect_case(const std::string& case_base, int tolerance_lsb) {
  const std::vector<std::uint32_t> meta = read_scalar_memh_u32(case_base + ".meta.memh");
  const std::vector<std::int32_t> x_chunks = read_scalar_memh_i32(case_base + ".core_x_chunks.memh");
  const std::vector<std::int32_t> y_expected = read_scalar_memh_i32(case_base + ".core_y_chunks.memh");

  const int elem_count = static_cast<int>(meta.at(1));
  const int chunk_count = (elem_count + 31) / 32;

  stream_t<hidden_vec32_t> x_stream;
  stream_t<hidden_vec32_t> y_stream;

  for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
    hidden_vec32_t chunk{};
    for (int elem = 0; elem < 32; ++elem) {
      chunk.data[elem] = q16_to_fixed(x_chunks[(chunk_idx * 32) + elem]);
    }
    x_stream.write(chunk);
  }

  silu_core_hls(
      x_stream,
      y_stream,
      static_cast<std::uint16_t>(elem_count));

  for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
    if (y_stream.empty()) {
      std::cerr << "SiLU output underflow for " << case_base << "\n";
      std::exit(1);
    }
    hidden_vec32_t chunk = y_stream.read();
    for (int elem = 0; elem < 32; ++elem) {
      const std::int32_t expected = y_expected[(chunk_idx * 32) + elem];
      const std::int32_t actual = fixed_to_q16(chunk.data[elem]);
      if (std::abs(actual - expected) > tolerance_lsb) {
        std::cerr << "SiLU mismatch for " << case_base
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
  expect_case("sim/golden_traces/phase5/rtl/phase5_prefill_layer0_silu_gate_m0", 32);
  expect_case("sim/golden_traces/phase5/rtl/phase5_decode_layer0_silu_gate_m0", 32);
  std::cout << "PASS: tb_silu\n";
  return 0;
}
