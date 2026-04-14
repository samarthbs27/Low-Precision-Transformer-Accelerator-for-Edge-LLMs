#include "rmsnorm_core_hls.hpp"

#include "../common/nonlinear_math.hpp"
#include "../common/stream_utils.hpp"

namespace tinyllama {
namespace rmsnorm {

using namespace hls_common;

void rmsnorm_core_hls(
    stream_t<hidden_vec32_t>& act_chunks,
    stream_t<hidden_vec32_t>& gamma_chunks,
    stream_t<hidden_vec32_t>& out_chunks,
    std::uint16_t row_count,
    std::uint16_t feature_count,
    fixed_t epsilon) {
  const int feature_chunks = feature_count / kChunkElems;

  hidden_vec32_t gamma_buf[kRmsNormFeatureChunks];
  hidden_vec32_t act_buf[kSeqTile][kRmsNormFeatureChunks];
  accum_fixed_t sumsq[kSeqTile];
  fixed_t inv_rms[kSeqTile];

  for (int row = 0; row < kSeqTile; ++row) {
    sumsq[row] = static_cast<accum_fixed_t>(0);
    inv_rms[row] = static_cast<fixed_t>(0);
  }

  for (int feature_chunk = 0; feature_chunk < feature_chunks; ++feature_chunk) {
    gamma_buf[feature_chunk] = read_blocking(gamma_chunks);
  }

  for (int feature_chunk = 0; feature_chunk < feature_chunks; ++feature_chunk) {
    for (int row = 0; row < row_count; ++row) {
      hidden_vec32_t chunk = read_blocking(act_chunks);
      act_buf[row][feature_chunk] = chunk;
      for (int elem = 0; elem < kChunkElems; ++elem) {
        accum_fixed_t sample = static_cast<accum_fixed_t>(chunk.data[elem]);
        sumsq[row] += sample * sample;
      }
    }
  }

  for (int row = 0; row < row_count; ++row) {
    fixed_t mean_sq = static_cast<fixed_t>(sumsq[row]) / static_cast<fixed_t>(feature_count);
    inv_rms[row] = static_cast<fixed_t>(1) / fixed_sqrt(mean_sq + epsilon);
  }

  for (int feature_chunk = 0; feature_chunk < feature_chunks; ++feature_chunk) {
    for (int row = 0; row < row_count; ++row) {
      hidden_vec32_t out_chunk;
      for (int elem = 0; elem < kChunkElems; ++elem) {
        out_chunk.data[elem] =
            act_buf[row][feature_chunk].data[elem] * inv_rms[row] * gamma_buf[feature_chunk].data[elem];
      }
      write_blocking(out_chunks, out_chunk);
    }
  }
}

}  // namespace rmsnorm
}  // namespace tinyllama
