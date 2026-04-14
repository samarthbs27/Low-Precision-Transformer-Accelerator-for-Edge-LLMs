#ifndef TINYLLAMA_HLS_SILU_CORE_HLS_HPP_
#define TINYLLAMA_HLS_SILU_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace tinyllama {
namespace silu {

using hls_common::hidden_vec32_t;
using hls_common::stream_t;

void silu_core_hls(
    stream_t<hidden_vec32_t>& in_chunks,
    stream_t<hidden_vec32_t>& out_chunks,
    std::uint16_t elem_count);

}  // namespace silu
}  // namespace tinyllama

#endif  // TINYLLAMA_HLS_SILU_CORE_HLS_HPP_
