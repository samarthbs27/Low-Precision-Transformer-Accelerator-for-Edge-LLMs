#include "softmax_core_hls.hpp"

#include "../common/nonlinear_math.hpp"
#include "../common/stream_utils.hpp"

namespace tinyllama {
namespace softmax {

using namespace hls_common;

void softmax_core_hls(
    stream_t<hidden_vec32_t>& score_chunks,
    stream_t<hidden_vec32_t>& prob_chunks,
    std::uint16_t row_count,
    std::uint16_t key_col_count) {
  fixed_t score_buf[kScoreRowsPerChunk][kScoreKTile];

  for (int row = 0; row < kScoreRowsPerChunk; ++row) {
    for (int col = 0; col < kScoreKTile; ++col) {
      score_buf[row][col] = static_cast<fixed_t>(0);
    }
  }

  for (int row = 0; row < row_count; ++row) {
    for (int chunk = 0; chunk < (kScoreKTile / kChunkElems); ++chunk) {
      hidden_vec32_t in_chunk = read_blocking(score_chunks);
      for (int elem = 0; elem < kChunkElems; ++elem) {
        const int col = (chunk * kChunkElems) + elem;
        score_buf[row][col] = in_chunk.data[elem];
      }
    }
  }

  for (int row = 0; row < kScoreRowsPerChunk; ++row) {
    if (row < row_count) {
      fixed_t row_max = score_buf[row][0];
      for (int col = 1; col < key_col_count; ++col) {
        row_max = max_value(row_max, score_buf[row][col]);
      }

      fixed_t exp_buf[kScoreKTile];
      fixed_t sum_exp = static_cast<fixed_t>(0);
      for (int col = 0; col < key_col_count; ++col) {
        exp_buf[col] = fixed_exp(score_buf[row][col] - row_max);
        sum_exp += exp_buf[col];
      }
      for (int col = key_col_count; col < kScoreKTile; ++col) {
        exp_buf[col] = static_cast<fixed_t>(0);
      }

      for (int chunk = 0; chunk < (kScoreKTile / kChunkElems); ++chunk) {
        hidden_vec32_t out_chunk;
        for (int elem = 0; elem < kChunkElems; ++elem) {
          const int col = (chunk * kChunkElems) + elem;
          out_chunk.data[elem] =
              (col < key_col_count) ? (exp_buf[col] / sum_exp) : static_cast<fixed_t>(0);
        }
        write_blocking(prob_chunks, out_chunk);
      }
    } else {
      for (int chunk = 0; chunk < (kScoreKTile / kChunkElems); ++chunk) {
        hidden_vec32_t out_chunk;
        zero_vec(&out_chunk);
        write_blocking(prob_chunks, out_chunk);
      }
    }
  }
}

}  // namespace softmax
}  // namespace tinyllama
