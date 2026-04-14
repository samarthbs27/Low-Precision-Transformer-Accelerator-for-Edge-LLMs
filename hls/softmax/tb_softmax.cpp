#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../common/fixed_types.hpp"
#include "../common/test_memh.hpp"
#include "softmax_core_hls.hpp"

namespace {

using tinyllama::hls_common::fixed_t;
using tinyllama::hls_common::hidden_vec32_t;
using tinyllama::hls_common::read_scalar_memh_i32;
using tinyllama::hls_common::read_scalar_memh_u32;
using tinyllama::hls_common::stream_t;
using tinyllama::softmax::softmax_core_hls;

fixed_t q16_to_fixed(std::int32_t raw_q16) {
  return static_cast<fixed_t>(static_cast<float>(raw_q16) / 65536.0f);
}

std::int32_t fixed_to_q16(fixed_t value) {
  return static_cast<std::int32_t>(std::lrint(static_cast<float>(value) * 65536.0f));
}

void expect_case(const std::string& case_base, int tolerance_lsb) {
  const std::vector<std::uint32_t> meta = read_scalar_memh_u32(case_base + ".meta.memh");
  const std::vector<std::int32_t> score_chunks = read_scalar_memh_i32(case_base + ".core_score_chunks.memh");
  const std::vector<std::int32_t> prob_expected = read_scalar_memh_i32(case_base + ".core_prob_chunks.memh");

  const int row_count = static_cast<int>(meta.at(0));
  const int chunk_count = 16;

  stream_t<hidden_vec32_t> score_stream;
  stream_t<hidden_vec32_t> prob_stream;

  for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
    hidden_vec32_t chunk{};
    for (int elem = 0; elem < 32; ++elem) {
      chunk.data[elem] = q16_to_fixed(score_chunks[(chunk_idx * 32) + elem]);
    }
    score_stream.write(chunk);
  }

  softmax_core_hls(
      score_stream,
      prob_stream,
      static_cast<std::uint16_t>(row_count),
      static_cast<std::uint16_t>(64));

  for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
    if (prob_stream.empty()) {
      std::cerr << "Softmax output underflow for " << case_base << "\n";
      std::exit(1);
    }
    hidden_vec32_t chunk = prob_stream.read();
    for (int elem = 0; elem < 32; ++elem) {
      const std::int32_t expected = prob_expected[(chunk_idx * 32) + elem];
      const std::int32_t actual = fixed_to_q16(chunk.data[elem]);
      if (std::abs(actual - expected) > tolerance_lsb) {
        std::cerr << "Softmax mismatch for " << case_base
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
  expect_case("sim/golden_traces/phase5/rtl/phase5_prefill_layer0_softmax_q0_kv0_qb8_kb0", 32);
  expect_case("sim/golden_traces/phase5/rtl/phase5_decode_layer0_softmax_q0_kv0_qb15_kb0", 32);
  std::cout << "PASS: tb_softmax\n";
  return 0;
}
