import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tinyllama_u55c_shell_wrapper (
  input  logic                        ap_clk,
  input  logic                        ap_rst_n,
  input  logic [AXIL_ADDR_W-1:0]      s_axi_awaddr,
  input  logic                        s_axi_awvalid,
  output logic                        s_axi_awready,
  input  logic [AXIL_DATA_W-1:0]      s_axi_wdata,
  input  logic [AXIL_STRB_W-1:0]      s_axi_wstrb,
  input  logic                        s_axi_wvalid,
  output logic                        s_axi_wready,
  output logic [1:0]                  s_axi_bresp,
  output logic                        s_axi_bvalid,
  input  logic                        s_axi_bready,
  input  logic [AXIL_ADDR_W-1:0]      s_axi_araddr,
  input  logic                        s_axi_arvalid,
  output logic                        s_axi_arready,
  output logic [AXIL_DATA_W-1:0]      s_axi_rdata,
  output logic [1:0]                  s_axi_rresp,
  output logic                        s_axi_rvalid,
  input  logic                        s_axi_rready,
  output logic                        interrupt,
  output logic                        shell_rd_desc_valid_o,
  input  logic                        shell_rd_desc_ready_i,
  output dma_desc_t                   shell_rd_desc_o,
  input  logic                        shell_rd_data_valid_i,
  output logic                        shell_rd_data_ready_o,
  input  logic [DMA_BEAT_W-1:0]       shell_rd_data_i,
  output logic                        shell_wr_desc_valid_o,
  input  logic                        shell_wr_desc_ready_i,
  output dma_desc_t                   shell_wr_desc_o,
  output logic                        shell_wr_data_valid_o,
  input  logic                        shell_wr_data_ready_i,
  output logic [DMA_BEAT_W-1:0]       shell_wr_data_o
);

  localparam int unsigned DMA_DESC_W = $bits(dma_desc_t);
  localparam int unsigned DMA_WR_REQ_W = DMA_DESC_W + DMA_BEAT_W;

  logic                     core_rd_desc_valid;
  logic                     core_rd_desc_ready;
  dma_desc_t                core_rd_desc;
  logic [DMA_DESC_W-1:0]    core_rd_desc_packed;
  logic [DMA_DESC_W-1:0]    shell_rd_desc_packed;

  logic                     core_rd_data_valid;
  logic                     core_rd_data_ready;
  logic [DMA_BEAT_W-1:0]    core_rd_data;

  logic                     core_wr_desc_valid;
  logic                     core_wr_req_ready;
  dma_desc_t                core_wr_desc;
  logic [DMA_DESC_W-1:0]    core_wr_desc_packed;
  logic                     core_wr_data_valid;
  logic [DMA_BEAT_W-1:0]    core_wr_data;
  logic [DMA_WR_REQ_W-1:0]  core_wr_req_packed;
  logic [DMA_WR_REQ_W-1:0]  shell_wr_req_packed;
  logic                     shell_wr_req_valid;
  logic                     shell_wr_req_ready;
  logic [DMA_DESC_W-1:0]    shell_wr_desc_packed;

  assign core_rd_desc_packed = core_rd_desc;
  assign shell_rd_desc_o     = dma_desc_t'(shell_rd_desc_packed);
`ifndef SYNTHESIS
  assign core_wr_desc_packed = core_wr_desc;
`endif
  assign core_wr_req_packed  = {core_wr_desc_packed, core_wr_data};
  assign shell_wr_desc_packed = shell_wr_req_packed[DMA_WR_REQ_W-1 -: DMA_DESC_W];
  assign shell_wr_desc_o     = dma_desc_t'(shell_wr_desc_packed);
  assign shell_wr_data_o     = shell_wr_req_packed[DMA_BEAT_W-1:0];
  assign shell_wr_desc_valid_o = shell_wr_req_valid;
  assign shell_wr_data_valid_o = shell_wr_req_valid;
  assign shell_wr_req_ready    = shell_wr_desc_ready_i && shell_wr_data_ready_i;

  tinyllama_u55c_kernel_top u_tinyllama_u55c_kernel_top (
    .ap_clk              (ap_clk),
    .ap_rst_n            (ap_rst_n),
    .s_axi_awaddr        (s_axi_awaddr),
    .s_axi_awvalid       (s_axi_awvalid),
    .s_axi_awready       (s_axi_awready),
    .s_axi_wdata         (s_axi_wdata),
    .s_axi_wstrb         (s_axi_wstrb),
    .s_axi_wvalid        (s_axi_wvalid),
    .s_axi_wready        (s_axi_wready),
    .s_axi_bresp         (s_axi_bresp),
    .s_axi_bvalid        (s_axi_bvalid),
    .s_axi_bready        (s_axi_bready),
    .s_axi_araddr        (s_axi_araddr),
    .s_axi_arvalid       (s_axi_arvalid),
    .s_axi_arready       (s_axi_arready),
    .s_axi_rdata         (s_axi_rdata),
    .s_axi_rresp         (s_axi_rresp),
    .s_axi_rvalid        (s_axi_rvalid),
    .s_axi_rready        (s_axi_rready),
    .interrupt           (interrupt),
    .shell_rd_desc_valid_o(core_rd_desc_valid),
    .shell_rd_desc_ready_i(core_rd_desc_ready),
    .shell_rd_desc_o     (core_rd_desc),
    .shell_rd_data_valid_i(core_rd_data_valid),
    .shell_rd_data_ready_o(core_rd_data_ready),
    .shell_rd_data_i     (core_rd_data),
    .shell_wr_desc_valid_o(core_wr_desc_valid),
    .shell_wr_desc_ready_i(core_wr_req_ready),
    .shell_wr_desc_o     (core_wr_desc),
    .shell_wr_data_valid_o(core_wr_data_valid),
    .shell_wr_data_ready_i(core_wr_req_ready),
    .shell_wr_data_o     (core_wr_data)
  );

  skid_buffer #(
    .DATA_W(DMA_DESC_W)
  ) u_shell_rd_desc_skid (
    .clk      (ap_clk),
    .rst_n    (ap_rst_n),
    .in_data  (core_rd_desc_packed),
    .in_valid (core_rd_desc_valid),
    .in_ready (core_rd_desc_ready),
    .out_data (shell_rd_desc_packed),
    .out_valid(shell_rd_desc_valid_o),
    .out_ready(shell_rd_desc_ready_i),
    .occupancy()
  );

  skid_buffer #(
    .DATA_W(DMA_BEAT_W)
  ) u_shell_rd_data_skid (
    .clk      (ap_clk),
    .rst_n    (ap_rst_n),
    .in_data  (shell_rd_data_i),
    .in_valid (shell_rd_data_valid_i),
    .in_ready (shell_rd_data_ready_o),
    .out_data (core_rd_data),
    .out_valid(core_rd_data_valid),
    .out_ready(core_rd_data_ready),
    .occupancy()
  );

  skid_buffer #(
    .DATA_W(DMA_WR_REQ_W)
  ) u_shell_wr_req_skid (
    .clk      (ap_clk),
    .rst_n    (ap_rst_n),
    .in_data  (core_wr_req_packed),
    .in_valid (core_wr_desc_valid && core_wr_data_valid),
    .in_ready (core_wr_req_ready),
    .out_data (shell_wr_req_packed),
    .out_valid(shell_wr_req_valid),
    .out_ready(shell_wr_req_ready),
    .occupancy()
  );

endmodule
