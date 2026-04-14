#ifndef TINYLLAMA_HLS_COMMON_FIXED_TYPES_HPP_
#define TINYLLAMA_HLS_COMMON_FIXED_TYPES_HPP_

#include <cstdint>
#include <queue>
#include <type_traits>

#if __has_include(<ap_fixed.h>) && __has_include(<ap_int.h>) && __has_include(<hls_stream.h>)
  #include <ap_fixed.h>
  #include <ap_int.h>
  #include <hls_stream.h>
  #define TINYLLAMA_HLS_NATIVE_TYPES 1
#else
  #define TINYLLAMA_HLS_NATIVE_TYPES 0

template <int W, int I>
using ap_fixed = float;

template <int W>
using ap_int = std::conditional_t<(W <= 8),  std::int8_t,
               std::conditional_t<(W <= 16), std::int16_t,
               std::conditional_t<(W <= 32), std::int32_t,
                                  std::int64_t>>>;

template <int W>
using ap_uint = std::conditional_t<(W <= 8),  std::uint8_t,
                std::conditional_t<(W <= 16), std::uint16_t,
                std::conditional_t<(W <= 32), std::uint32_t,
                                   std::uint64_t>>>;

namespace hls {
template <typename T>
class stream {
 public:
  stream() = default;

  void write(const T& value) { queue_.push(value); }

  T read() {
    T value = queue_.front();
    queue_.pop();
    return value;
  }

  bool empty() const { return queue_.empty(); }

 private:
  std::queue<T> queue_;
};
}  // namespace hls

#endif

namespace tinyllama {
namespace hls_common {

constexpr int kNLayers      = 22;
constexpr int kDModel       = 2048;
constexpr int kDFF          = 5632;
constexpr int kNQHeads      = 32;
constexpr int kNKVHeads     = 4;
constexpr int kHeadDim      = 64;
constexpr int kSeqTile      = 64;
constexpr int kVocabTile    = 128;
constexpr int kTileBanks    = 16;
constexpr int kBankSlice8   = 32;
constexpr int kChunkElems   = 32;
constexpr int kDmaBeatBits  = 256;
constexpr int kFixedTotalW  = 32;
constexpr int kFixedIntW    = 16;
constexpr int kFixedFracW   = kFixedTotalW - kFixedIntW;
constexpr int kProbScale    = 127;
constexpr int kRmsNormFeatureChunks = kDModel / kChunkElems;
constexpr int kActTileChunks = kSeqTile * kDModel / kChunkElems;
constexpr int kScoreRowsPerChunk = 8;
constexpr int kScoreKTile = 64;
constexpr int kSoftmaxChunkElems = 8 * 64;
constexpr int kSoftmaxChunksPerTile = kSoftmaxChunkElems / kChunkElems;
constexpr int kTileElems = 512;
constexpr int kTileChunks = kTileElems / kChunkElems;

using fixed_t       = ap_fixed<kFixedTotalW, kFixedIntW>;
using accum_fixed_t = ap_fixed<48, 24>;
using score_fixed_t = ap_fixed<32, 16>;
using prob_fixed_t  = ap_fixed<32, 16>;

using act_int8_t    = ap_int<8>;
using wt_int8_t     = ap_int<8>;
using prob_int8_t   = ap_uint<8>;
using accum_int32_t = ap_int<32>;
using scale_int32_t = ap_int<32>;
using token_t       = ap_uint<32>;

template <typename T>
using stream_t = hls::stream<T>;

template <typename T, int N>
struct vec_t {
  T data[N];
};

using hidden_vec32_t = vec_t<fixed_t, 32>;
using hidden_vec64_t = vec_t<fixed_t, 64>;
using int8_vec32_t   = vec_t<act_int8_t, 32>;
using int8_vec64_t   = vec_t<act_int8_t, 64>;
using scale_vec16_t  = vec_t<scale_int32_t, 16>;
using prob_vec32_t   = vec_t<prob_fixed_t, 32>;

}  // namespace hls_common
}  // namespace tinyllama

#endif  // TINYLLAMA_HLS_COMMON_FIXED_TYPES_HPP_
