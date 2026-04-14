#include "silu_core_hls.hpp"

#include "../common/nonlinear_math.hpp"
#include "../common/stream_utils.hpp"

namespace tinyllama {
namespace silu {

using namespace hls_common;

void silu_core_hls(
    stream_t<hidden_vec32_t>& in_chunks,
    stream_t<hidden_vec32_t>& out_chunks,
    std::uint16_t elem_count) {
  const int active_chunks = (elem_count + kChunkElems - 1) / kChunkElems;

  for (int chunk_idx = 0; chunk_idx < active_chunks; ++chunk_idx) {
    hidden_vec32_t in_chunk = read_blocking(in_chunks);
    hidden_vec32_t out_chunk;
    for (int elem = 0; elem < kChunkElems; ++elem) {
      const int flat_idx = (chunk_idx * kChunkElems) + elem;
      if (flat_idx < elem_count) {
        const fixed_t x = in_chunk.data[elem];
        out_chunk.data[elem] = x * fixed_sigmoid(x);
      } else {
        out_chunk.data[elem] = static_cast<fixed_t>(0);
      }
    }
    write_blocking(out_chunks, out_chunk);
  }
}

}  // namespace silu
}  // namespace tinyllama
