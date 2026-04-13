import tinyllama_pkg::*;

module axi_lite_ctrl_slave (
  input  logic                         ap_clk,
  input  logic                         ap_rst_n,
  input  logic [AXIL_ADDR_W-1:0]       s_axi_awaddr,
  input  logic                         s_axi_awvalid,
  output logic                         s_axi_awready,
  input  logic [AXIL_DATA_W-1:0]       s_axi_wdata,
  input  logic [AXIL_STRB_W-1:0]       s_axi_wstrb,
  input  logic                         s_axi_wvalid,
  output logic                         s_axi_wready,
  output logic [1:0]                   s_axi_bresp,
  output logic                         s_axi_bvalid,
  input  logic                         s_axi_bready,
  input  logic [AXIL_ADDR_W-1:0]       s_axi_araddr,
  input  logic                         s_axi_arvalid,
  output logic                         s_axi_arready,
  output logic [AXIL_DATA_W-1:0]       s_axi_rdata,
  output logic [1:0]                   s_axi_rresp,
  output logic                         s_axi_rvalid,
  input  logic                         s_axi_rready,
  output logic                         reg_wr_en,
  output logic [REG_WORD_ADDR_W-1:0]   reg_wr_addr,
  output logic [AXIL_DATA_W-1:0]       reg_wr_data,
  output logic [AXIL_STRB_W-1:0]       reg_wr_strb,
  output logic                         reg_rd_en,
  output logic [REG_WORD_ADDR_W-1:0]   reg_rd_addr,
  input  logic [AXIL_DATA_W-1:0]       reg_rd_data
);

  logic [AXIL_ADDR_W-1:0] awaddr_q;
  logic                   aw_pending_q;
  logic [AXIL_DATA_W-1:0] wdata_q;
  logic [AXIL_STRB_W-1:0] wstrb_q;
  logic                   w_pending_q;
  logic                   rd_pending_q;

  assign s_axi_awready = !aw_pending_q && !s_axi_bvalid;
  assign s_axi_wready  = !w_pending_q  && !s_axi_bvalid;
  assign s_axi_bresp   = 2'b00;
  assign s_axi_arready = !rd_pending_q && !s_axi_rvalid;
  assign s_axi_rresp   = 2'b00;

  always_ff @(posedge ap_clk) begin
    logic aw_fire;
    logic w_fire;
    logic [AXIL_ADDR_W-1:0] awaddr_sel;
    logic [AXIL_DATA_W-1:0] wdata_sel;
    logic [AXIL_STRB_W-1:0] wstrb_sel;

    aw_fire   = s_axi_awvalid && s_axi_awready;
    w_fire    = s_axi_wvalid  && s_axi_wready;
    awaddr_sel = aw_fire ? s_axi_awaddr : awaddr_q;
    wdata_sel  = w_fire ? s_axi_wdata   : wdata_q;
    wstrb_sel  = w_fire ? s_axi_wstrb   : wstrb_q;

    reg_wr_en   <= 1'b0;
    reg_rd_en   <= 1'b0;

    if (!ap_rst_n) begin
      awaddr_q     <= '0;
      aw_pending_q <= 1'b0;
      wdata_q      <= '0;
      wstrb_q      <= '0;
      w_pending_q  <= 1'b0;
      s_axi_bvalid <= 1'b0;
      reg_wr_addr  <= '0;
      reg_wr_data  <= '0;
      reg_wr_strb  <= '0;
      rd_pending_q <= 1'b0;
      reg_rd_addr  <= '0;
      s_axi_rdata  <= '0;
      s_axi_rvalid <= 1'b0;
    end else begin
      if (aw_fire) begin
        awaddr_q     <= s_axi_awaddr;
        aw_pending_q <= 1'b1;
      end

      if (w_fire) begin
        wdata_q     <= s_axi_wdata;
        wstrb_q     <= s_axi_wstrb;
        w_pending_q <= 1'b1;
      end

      if ((aw_pending_q || aw_fire) && (w_pending_q || w_fire) && !s_axi_bvalid) begin
        reg_wr_en   <= 1'b1;
        reg_wr_addr <= awaddr_sel[AXIL_ADDR_W-1:2];
        reg_wr_data <= wdata_sel;
        reg_wr_strb <= wstrb_sel;
        aw_pending_q <= 1'b0;
        w_pending_q  <= 1'b0;
        s_axi_bvalid <= 1'b1;
      end

      if (s_axi_bvalid && s_axi_bready) begin
        s_axi_bvalid <= 1'b0;
      end

      if (s_axi_arvalid && s_axi_arready) begin
        reg_rd_en    <= 1'b1;
        reg_rd_addr  <= s_axi_araddr[AXIL_ADDR_W-1:2];
        rd_pending_q <= 1'b1;
      end

      if (rd_pending_q) begin
        s_axi_rdata  <= reg_rd_data;
        s_axi_rvalid <= 1'b1;
        rd_pending_q <= 1'b0;
      end

      if (s_axi_rvalid && s_axi_rready) begin
        s_axi_rvalid <= 1'b0;
      end
    end
  end

endmodule
