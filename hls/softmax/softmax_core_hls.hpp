#ifndef TINYLLAMA_HLS_SOFTMAX_CORE_HLS_HPP_
#define TINYLLAMA_HLS_SOFTMAX_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace tinyllama {
namespace softmax {

using hls_common::fixed_t;
using hls_common::hidden_vec32_t;
using hls_common::stream_t;

// Chunk ordering is row-major over one fixed 8x64 score chunk:
//   for row in [0, row_count):
//     for chunk in [0, 2):
//       emit one 32-element chunk.
void softmax_core_hls(
    stream_t<hidden_vec32_t>& score_chunks,
    stream_t<hidden_vec32_t>& prob_chunks,
    std::uint16_t row_count,
    std::uint16_t key_col_count);

}  // namespace softmax
}  // namespace tinyllama

#endif  // TINYLLAMA_HLS_SOFTMAX_CORE_HLS_HPP_
