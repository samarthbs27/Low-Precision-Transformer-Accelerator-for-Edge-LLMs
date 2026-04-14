# RTL Testbenches

This folder contains production TinyLlama RTL smoke tests for the new `rtl/common/` building blocks.

## Files

| File | What it tests | Smoke test |
|------|----------------|------------|
| `tb_stream_fifo.sv` | Directed self-check of `rtl/common/stream_fifo.sv`, including push/pop ordering, occupancy tracking, simultaneous push/pop, full-condition backpressure, and blocked-push behavior. | Run the `tb_stream_fifo` command below. |
| `tb_descriptor_fifo.sv` | Directed self-check of `rtl/common/descriptor_fifo.sv`, including push, pop, and occupancy behavior. | Run the `tb_descriptor_fifo` command below. |
| `tb_axi_lite_ctrl_slave.sv` | AXI-Lite plus register-file smoke test covering register writes, register reads, sticky status, clear-on-START behavior, launch mode, and abort behavior. | Run the `tb_axi_lite_ctrl_slave` command below. |
| `tb_host_cmd_status_mgr.sv` | Fake-HBM smoke test for one-beat command fetch from PC30, launch/busy status writeback, terminal status writeback, and relaunch/error-only command-status behavior. | Run the `tb_host_cmd_status_mgr` command below. |
| `tb_prefill_decode_controller.sv` | Control-path smoke test covering command-aware prefill launch, prompt-read wait, 22-layer iteration, per-layer block progression through the reused layer controller, LM-head/token handoff, EOS stop, MAX_TOKENS stop, host-abort stop, decode-only launch, and zero-token prefill rejection. | Run the `tb_prefill_decode_controller` command below. |
| `tb_hbm_port_router.sv` | Directed arbitration/routing smoke test for the fixed-function HBM router. | Run the `tb_hbm_port_router` command below. |
| `tb_tile_buffer_bank.sv` | Directed ping/pong banked storage smoke test for the generic tile buffer. | Run the `tb_tile_buffer_bank` command below. |
| `tb_prompt_token_reader.sv` | Directed host-I/O DMA smoke test for prompt token burst fetch and token stream emission. | Run the `tb_prompt_token_reader` command below. |
| `tb_generated_token_writer.sv` | Directed host-I/O DMA smoke test for generated-token write beats and ring-buffer wrap behavior. | Run the `tb_generated_token_writer` command below. |
| `tb_scale_metadata_store.sv` | Directed read/write smoke test for multi-port scale metadata storage. | Run the `tb_scale_metadata_store` command below. |
| `tb_kv_cache_manager.sv` | Directed descriptor-generation smoke test for fixed KV-cache address and channel mapping. | Run the `tb_kv_cache_manager` command below. |
| `tb_weight_dma_reader.sv` | Directed multi-beat weight-reader smoke test covering descriptor handshake, streamed beat output, and tensor-to-block tag mapping. | Run the `tb_weight_dma_reader` command below. |
| `tb_kv_cache_dma_reader.sv` | Directed multi-beat KV-cache reader smoke test covering descriptor handshake, V/K block tagging, and final-beat marking. | Run the `tb_kv_cache_dma_reader` command below. |
| `tb_kv_cache_dma_writer.sv` | Directed KV-cache write smoke test covering buffered payload capture and ready/valid behavior. | Run the `tb_kv_cache_dma_writer` command below. |
| `tb_embedding_lmhead_dma_reader.sv` | Directed multi-mode reader smoke test for embedding beat streams, LM-head weight beats, and aggregated scale metadata. | Run the `tb_embedding_lmhead_dma_reader` command below. |
| `tb_mac_lane.sv` | Directed arithmetic smoke test for the signed INT8xINT8->INT32 MAC leaf. | Run the `tb_mac_lane` command below. |
| `tb_accumulator_bank.sv` | Directed stateful smoke test for 512-lane accumulator clear/load/tag behavior. | Run the `tb_accumulator_bank` command below. |
| `tb_requantize_unit.sv` | Directed arithmetic smoke test plus exported Phase 3 TinyLlama trace checks for bank-scaled Q16.16 requantization, rounding, and clamp behavior. | Run the `tb_requantize_unit` command below. |
| `tb_shared_gemm_engine.sv` | Directed smoke test plus exported Phase 3 TinyLlama trace replay for shared-engine accumulation, snapshot emission, and output backpressure. | Run the `tb_shared_gemm_engine` command below. |
| `tb_gemm_operand_router.sv` | Directed multi-mode router smoke test for Q, score, and weighted-sum operand selection plus active-mode tag normalization. | Run the `tb_gemm_operand_router` command below. |
| `tb_gemm_result_router.sv` | Directed multi-mode router smoke test for quantized outputs, raw score outputs, raw LM-head outputs, and active-mode tag normalization. | Run the `tb_gemm_result_router` command below. |
| `tb_gemm_op_scheduler.sv` | Directed scheduler smoke test for the legacy full decoder-layer GEMM order, LM-head-only schedule, and the new production block-driven GEMM mode. | Run the `tb_gemm_op_scheduler` command below. |
| `tb_rope_unit.sv` | Directed identity check plus exported Phase 4 TinyLlama trace replay for the RoPE rotary datapath. | Run the `tb_rope_unit` command below. |
| `tb_gqa_router.sv` | Directed grouped-query routing smoke test for K-path, V-path, KV-head validation, and select-conflict handling. | Run the `tb_gqa_router` command below. |
| `tb_causal_mask_unit.sv` | Directed mask check plus exported Phase 4 prefill/decode trace replay for the pre-softmax causal-mask leaf. | Run the `tb_causal_mask_unit` command below. |
| `tb_rmsnorm_wrapper.sv` | Exported Phase 5 trace-backed smoke test for the RTL wrapper around the fixed-point RMSNorm HLS core, including gamma-beat capture, scale emission, and quantized output verification. | Run the `tb_rmsnorm_wrapper` command below. |
| `tb_softmax_wrapper.sv` | Exported Phase 5 trace-backed smoke test for the RTL wrapper around the fixed-point softmax HLS core, including masked-score dequantization, probability-scale emission, and INT8 probability output verification. | Run the `tb_softmax_wrapper` command below. |
| `tb_silu_wrapper.sv` | Exported Phase 5 trace-backed smoke test for the RTL wrapper around the fixed-point SiLU HLS core, including input dequantization, output requantization, and scale emission. | Run the `tb_silu_wrapper` command below. |
| `tb_embedding_lookup.sv` | Directed plus exported Phase 6 trace-backed smoke test for embedding-row DMA request generation and full-row FP16 assembly. | Run the `tb_embedding_lookup` command below. |
| `tb_embedding_quantizer.sv` | Directed plus exported Phase 6 trace-backed smoke test for FP16 embedding-row batching, Q16.16 scale-based quantization, scale emission, and INT8 activation-tile output. | Run the `tb_embedding_quantizer` command below. |
| `tb_residual_add.sv` | Directed plus exported Phase 6 trace-backed smoke test for aligned residual1/residual2 INT32 accumulation and tag retiming. | Run the `tb_residual_add` command below. |
| `tb_elementwise_mul.sv` | Directed plus exported Phase 6 trace-backed smoke test for the SwiGLU `SiLU(gate) * up` multiply leaf. | Run the `tb_elementwise_mul` command below. |
| `tb_lm_head_controller.sv` | Directed controller smoke test for the outer LM-head vocab-tile loop and partial-logit retagging path. | Run the `tb_lm_head_controller` command below. |
| `tb_argmax_reduction.sv` | Directed plus exported Phase 6 trace-backed smoke test for full-vocabulary greedy argmax with tie-break verification. | Run the `tb_argmax_reduction` command below. |
| `tb_debug_capture_mux.sv` | Directed smoke test for layer/block filtering, lowest-index source priority, and drop-pulse behavior in the debug mux. | Run the `tb_debug_capture_mux` command below. |
| `tb_decoder_layer_smoke.sv` | Exported Phase 7 trace-backed smoke test for one concrete decoder-layer pass, including block order, head-loop sequencing, block-driven GEMM tile counts, and operand/result routing checks. | Run the `tb_decoder_layer_smoke` command below. |
| `tb_prefill_decode_smoke.sv` | Exported Phase 8 trace-backed runtime-control smoke for prompt prefill plus a few decode steps, including expected generated-token count, layer-pass count, and final stop reason. | Run the `tb_prefill_decode_smoke` command below. |
| `tb_kernel_top_smoke.sv` | Exported Phase 8 top-level smoke for AXI-Lite launch, host command fetch, prompt reads, generated-token writes, final status writeback, and interrupt behavior. | Run the `tb_kernel_top_smoke` command below. |

## Smoke Tests

Run all commands from the project root.

Place all generated simulator outputs and logs under `sim/`.

### `tb_stream_fifo.sv`

```powershell
iverilog -g2012 -o sim/tb_stream_fifo.vvp `
  rtl/common/stream_fifo.sv `
  rtl/tb/tb_stream_fifo.sv
vvp sim/tb_stream_fifo.vvp
```

Expected pass string:

```text
PASS: tb_stream_fifo
```

### `tb_descriptor_fifo.sv`

```powershell
iverilog -g2012 -o sim/tb_descriptor_fifo.vvp `
  rtl/common/stream_fifo.sv `
  rtl/common/descriptor_fifo.sv `
  rtl/tb/tb_descriptor_fifo.sv
vvp sim/tb_descriptor_fifo.vvp
```

Expected pass string:

```text
PASS: tb_descriptor_fifo
```

### `tb_axi_lite_ctrl_slave.sv`

```powershell
iverilog -g2012 -o sim/tb_axi_lite_ctrl_slave.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/control/axi_lite_ctrl_slave.sv `
  rtl/control/kernel_reg_file.sv `
  rtl/tb/tb_axi_lite_ctrl_slave.sv
vvp sim/tb_axi_lite_ctrl_slave.vvp
```

Expected pass string:

```text
PASS: tb_axi_lite_ctrl_slave
```

### `tb_host_cmd_status_mgr.sv`

```powershell
iverilog -g2012 -o sim/tb_host_cmd_status_mgr.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/control/host_cmd_status_mgr.sv `
  rtl/tb/tb_host_cmd_status_mgr.sv
vvp sim/tb_host_cmd_status_mgr.vvp
```

Expected pass string:

```text
PASS: tb_host_cmd_status_mgr
```

### `tb_prefill_decode_controller.sv`

```powershell
iverilog -g2012 -o sim/tb_prefill_decode_controller.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/control/prefill_decode_controller.sv `
  rtl/control/layer_controller.sv `
  rtl/control/stop_condition_unit.sv `
  rtl/tb/tb_prefill_decode_controller.sv
vvp sim/tb_prefill_decode_controller.vvp
```

Expected pass string:

```text
PASS: tb_prefill_decode_controller
```

### `tb_hbm_port_router.sv`

```powershell
iverilog -g2012 -o sim/tb_hbm_port_router.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/hbm_port_router.sv `
  rtl/tb/tb_hbm_port_router.sv
vvp sim/tb_hbm_port_router.vvp
```

Expected pass string:

```text
PASS: tb_hbm_port_router
```

### `tb_tile_buffer_bank.sv`

```powershell
iverilog -g2012 -o sim/tb_tile_buffer_bank.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/memory/tile_buffer_bank.sv `
  rtl/tb/tb_tile_buffer_bank.sv
vvp sim/tb_tile_buffer_bank.vvp
```

Expected pass string:

```text
PASS: tb_tile_buffer_bank
```

### `tb_prompt_token_reader.sv`

```powershell
iverilog -g2012 -o sim/tb_prompt_token_reader.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/prompt_token_reader.sv `
  rtl/tb/tb_prompt_token_reader.sv
vvp sim/tb_prompt_token_reader.vvp
```

Expected pass string:

```text
PASS: tb_prompt_token_reader
```

### `tb_generated_token_writer.sv`

```powershell
iverilog -g2012 -o sim/tb_generated_token_writer.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/generated_token_writer.sv `
  rtl/tb/tb_generated_token_writer.sv
vvp sim/tb_generated_token_writer.vvp
```

Expected pass string:

```text
PASS: tb_generated_token_writer
```

### `tb_scale_metadata_store.sv`

```powershell
iverilog -g2012 -o sim/tb_scale_metadata_store.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/scale_metadata_store.sv `
  rtl/tb/tb_scale_metadata_store.sv
vvp sim/tb_scale_metadata_store.vvp
```

Expected pass string:

```text
PASS: tb_scale_metadata_store
```

### `tb_kv_cache_manager.sv`

```powershell
iverilog -g2012 -o sim/tb_kv_cache_manager.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/kv_cache_manager.sv `
  rtl/tb/tb_kv_cache_manager.sv
vvp sim/tb_kv_cache_manager.vvp
```

Expected pass string:

```text
PASS: tb_kv_cache_manager
```

### `tb_weight_dma_reader.sv`

```powershell
iverilog -g2012 -o sim/tb_weight_dma_reader.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/weight_dma_reader.sv `
  rtl/tb/tb_weight_dma_reader.sv
vvp sim/tb_weight_dma_reader.vvp
```

Expected pass string:

```text
PASS: tb_weight_dma_reader
```

### `tb_kv_cache_dma_reader.sv`

```powershell
iverilog -g2012 -o sim/tb_kv_cache_dma_reader.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/kv_cache_dma_reader.sv `
  rtl/tb/tb_kv_cache_dma_reader.sv
vvp sim/tb_kv_cache_dma_reader.vvp
```

Expected pass string:

```text
PASS: tb_kv_cache_dma_reader
```

### `tb_kv_cache_dma_writer.sv`

```powershell
iverilog -g2012 -o sim/tb_kv_cache_dma_writer.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/kv_cache_dma_writer.sv `
  rtl/tb/tb_kv_cache_dma_writer.sv
vvp sim/tb_kv_cache_dma_writer.vvp
```

Expected pass string:

```text
PASS: tb_kv_cache_dma_writer
```

### `tb_embedding_lmhead_dma_reader.sv`

```powershell
iverilog -g2012 -o sim/tb_embedding_lmhead_dma_reader.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/memory/embedding_lmhead_dma_reader.sv `
  rtl/tb/tb_embedding_lmhead_dma_reader.sv
vvp sim/tb_embedding_lmhead_dma_reader.vvp
```

Expected pass string:

```text
PASS: tb_embedding_lmhead_dma_reader
```

### `tb_mac_lane.sv`

```powershell
iverilog -g2012 -o sim/tb_mac_lane.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/compute/mac_lane.sv `
  rtl/tb/tb_mac_lane.sv
vvp sim/tb_mac_lane.vvp
```

Expected pass string:

```text
PASS: tb_mac_lane
```

### `tb_accumulator_bank.sv`

```powershell
iverilog -g2012 -o sim/tb_accumulator_bank.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/accumulator_bank.sv `
  rtl/tb/tb_accumulator_bank.sv
vvp sim/tb_accumulator_bank.vvp
```

Expected pass string:

```text
PASS: tb_accumulator_bank
```

### `tb_requantize_unit.sv`

```powershell
python model/export_fpga_vectors.py --phase phase3 --layer 0 --output-dir sim/golden_traces
iverilog -g2012 -o sim/tb_requantize_unit.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/requantize_unit.sv `
  rtl/tb/tb_requantize_unit.sv
vvp sim/tb_requantize_unit.vvp
```

Expected pass string:

```text
PASS: tb_requantize_unit
```

### `tb_shared_gemm_engine.sv`

```powershell
python model/export_fpga_vectors.py --phase phase3 --layer 0 --output-dir sim/golden_traces
iverilog -g2012 -o sim/tb_shared_gemm_engine.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/accumulator_bank.sv `
  rtl/compute/shared_gemm_engine.sv `
  rtl/tb/tb_shared_gemm_engine.sv
vvp sim/tb_shared_gemm_engine.vvp
```

Expected pass string:

```text
PASS: tb_shared_gemm_engine
```

Note:

The exported-trace shared-engine smoke test is the slowest local Phase 3 test under Icarus and replays a real `K_TILE=64` q-projection slice from the TinyLlama reference path.

### `tb_gemm_operand_router.sv`

```powershell
iverilog -g2012 -o sim/tb_gemm_operand_router.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/gemm_operand_router.sv `
  rtl/tb/tb_gemm_operand_router.sv
vvp sim/tb_gemm_operand_router.vvp
```

Expected pass string:

```text
PASS: tb_gemm_operand_router
```

### `tb_gemm_result_router.sv`

```powershell
iverilog -g2012 -o sim/tb_gemm_result_router.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/requantize_unit.sv `
  rtl/compute/gemm_result_router.sv `
  rtl/tb/tb_gemm_result_router.sv
vvp sim/tb_gemm_result_router.vvp
```

Expected pass string:

```text
PASS: tb_gemm_result_router
```

### `tb_gemm_op_scheduler.sv`

```powershell
iverilog -g2012 -o sim/tb_gemm_op_scheduler.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/compute/gemm_op_scheduler.sv `
  rtl/tb/tb_gemm_op_scheduler.sv
vvp sim/tb_gemm_op_scheduler.vvp
```

Expected pass string:

```text
PASS: tb_gemm_op_scheduler
```

### `tb_decoder_layer_smoke.sv`

Before running this bench, regenerate the Phase 7 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase7 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_decoder_layer_smoke.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/control/layer_controller.sv `
  rtl/compute/requantize_unit.sv `
  rtl/compute/gemm_op_scheduler.sv `
  rtl/compute/gemm_operand_router.sv `
  rtl/compute/gemm_result_router.sv `
  rtl/tb/tb_decoder_layer_smoke.sv
vvp sim/tb_decoder_layer_smoke.vvp
```

Expected pass string:

```text
PASS: tb_decoder_layer_smoke
```

### `tb_rope_unit.sv`

Before running this bench, regenerate the Phase 4 fixtures and RoPE ROM files:

```powershell
python model/export_fpga_vectors.py --phase phase4 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_rope_unit.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/rope_lut_rom.sv `
  rtl/compute/rope_unit.sv `
  rtl/tb/tb_rope_unit.sv
vvp sim/tb_rope_unit.vvp
```

Expected pass string:

```text
PASS: tb_rope_unit
```

### `tb_gqa_router.sv`

```powershell
iverilog -g2012 -o sim/tb_gqa_router.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/gqa_router.sv `
  rtl/tb/tb_gqa_router.sv
vvp sim/tb_gqa_router.vvp
```

Expected pass string:

```text
PASS: tb_gqa_router
```

### `tb_causal_mask_unit.sv`

Before running this bench, regenerate the Phase 4 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase4 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_causal_mask_unit.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/causal_mask_unit.sv `
  rtl/tb/tb_causal_mask_unit.sv
vvp sim/tb_causal_mask_unit.vvp
```

Expected pass string:

```text
PASS: tb_causal_mask_unit
```

### `tb_rmsnorm_wrapper.sv`

Before running this bench, regenerate the Phase 5 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase5 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_rmsnorm_wrapper.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/nonlinear/rmsnorm_wrapper.sv `
  rtl/tb/tb_rmsnorm_wrapper.sv
vvp sim/tb_rmsnorm_wrapper.vvp
```

Expected pass string:

```text
PASS: tb_rmsnorm_wrapper
```

### `tb_softmax_wrapper.sv`

Before running this bench, regenerate the Phase 5 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase5 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_softmax_wrapper.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/nonlinear/softmax_wrapper.sv `
  rtl/tb/tb_softmax_wrapper.sv
vvp sim/tb_softmax_wrapper.vvp
```

Expected pass string:

```text
PASS: tb_softmax_wrapper
```

### `tb_silu_wrapper.sv`

Before running this bench, regenerate the Phase 5 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase5 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_silu_wrapper.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/nonlinear/silu_wrapper.sv `
  rtl/tb/tb_silu_wrapper.sv
vvp sim/tb_silu_wrapper.vvp
```

Expected pass string:

```text
PASS: tb_silu_wrapper
```

### `tb_embedding_lookup.sv`

Before running this bench, regenerate the Phase 6 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase6 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_embedding_lookup.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/embedding_lookup.sv `
  rtl/tb/tb_embedding_lookup.sv
vvp sim/tb_embedding_lookup.vvp
```

Expected pass string:

```text
PASS: tb_embedding_lookup
```

### `tb_embedding_quantizer.sv`

Before running this bench, regenerate the Phase 6 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase6 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_embedding_quantizer.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/embedding_quantizer.sv `
  rtl/tb/tb_embedding_quantizer.sv
vvp sim/tb_embedding_quantizer.vvp
```

Expected pass string:

```text
PASS: tb_embedding_quantizer
```

### `tb_residual_add.sv`

Before running this bench, regenerate the Phase 6 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase6 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_residual_add.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/residual_add.sv `
  rtl/tb/tb_residual_add.sv
vvp sim/tb_residual_add.vvp
```

Expected pass string:

```text
PASS: tb_residual_add
```

### `tb_elementwise_mul.sv`

Before running this bench, regenerate the Phase 6 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase6 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_elementwise_mul.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/elementwise_mul.sv `
  rtl/tb/tb_elementwise_mul.sv
vvp sim/tb_elementwise_mul.vvp
```

Expected pass string:

```text
PASS: tb_elementwise_mul
```

### `tb_lm_head_controller.sv`

```powershell
iverilog -g2012 -o sim/tb_lm_head_controller.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/lm_head_controller.sv `
  rtl/tb/tb_lm_head_controller.sv
vvp sim/tb_lm_head_controller.vvp
```

Expected pass string:

```text
PASS: tb_lm_head_controller
```

### `tb_argmax_reduction.sv`

Before running this bench, regenerate the Phase 6 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase6 --layer 0 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_argmax_reduction.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/argmax_reduction.sv `
  rtl/tb/tb_argmax_reduction.sv
vvp sim/tb_argmax_reduction.vvp
```

Expected pass string:

```text
PASS: tb_argmax_reduction
```

### `tb_debug_capture_mux.sv`

```powershell
iverilog -g2012 -o sim/tb_debug_capture_mux.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/compute/debug_capture_mux.sv `
  rtl/tb/tb_debug_capture_mux.sv
vvp sim/tb_debug_capture_mux.vvp
```

Expected pass string:

```text
PASS: tb_debug_capture_mux
```

### `tb_prefill_decode_smoke.sv`

Before running this bench, regenerate the Phase 8 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase8 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_prefill_decode_smoke.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/control/prefill_decode_controller.sv `
  rtl/control/layer_controller.sv `
  rtl/control/stop_condition_unit.sv `
  rtl/tb/tb_prefill_decode_smoke.sv
vvp sim/tb_prefill_decode_smoke.vvp
```

Expected pass string:

```text
PASS: tb_prefill_decode_smoke
```

### `tb_kernel_top_smoke.sv`

Before running this bench, regenerate the Phase 8 fixtures:

```powershell
python model/export_fpga_vectors.py --phase phase8 --output-dir sim/golden_traces
```

Then run:

```powershell
iverilog -g2012 -o sim/tb_kernel_top_smoke.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/common/stream_fifo.sv `
  rtl/common/skid_buffer.sv `
  rtl/common/descriptor_fifo.sv `
  rtl/control/axi_lite_ctrl_slave.sv `
  rtl/control/kernel_reg_file.sv `
  rtl/control/host_cmd_status_mgr.sv `
  rtl/control/prefill_decode_controller.sv `
  rtl/control/layer_controller.sv `
  rtl/control/stop_condition_unit.sv `
  rtl/memory/hbm_port_router.sv `
  rtl/memory/prompt_token_reader.sv `
  rtl/memory/generated_token_writer.sv `
  rtl/top/tinyllama_u55c_kernel_top.sv `
  rtl/tb/tb_kernel_top_smoke.sv
vvp sim/tb_kernel_top_smoke.vvp
```

Expected pass string:

```text
PASS: tb_kernel_top_smoke
```
