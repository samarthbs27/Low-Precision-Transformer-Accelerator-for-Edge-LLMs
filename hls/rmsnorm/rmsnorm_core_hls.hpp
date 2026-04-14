#ifndef TINYLLAMA_HLS_RMSNORM_CORE_HLS_HPP_
#define TINYLLAMA_HLS_RMSNORM_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace tinyllama {
namespace rmsnorm {

using hls_common::fixed_t;
using hls_common::hidden_vec32_t;
using hls_common::stream_t;

// Chunk ordering is feature-major, row-minor:
//   for feature_chunk in [0, feature_count / 32):
//     for row in [0, row_count):
//       emit one 32-element chunk.
void rmsnorm_core_hls(
    stream_t<hidden_vec32_t>& act_chunks,
    stream_t<hidden_vec32_t>& gamma_chunks,
    stream_t<hidden_vec32_t>& out_chunks,
    std::uint16_t row_count,
    std::uint16_t feature_count,
    fixed_t epsilon);

}  // namespace rmsnorm
}  // namespace tinyllama

#endif  // TINYLLAMA_HLS_RMSNORM_CORE_HLS_HPP_
