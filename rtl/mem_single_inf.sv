/*
 * Copyright (c) 2019, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
`timescale 1ns / 1ps
`default_nettype none

`include "example_module.vh"

module mem_single_inf #(
    parameter ENABLE = 1,
    parameter UNALIGNED = 0,
    parameter AXI_ID_WIDTH = 1
)(
    input wire                  user_clk,
    input wire                  user_aresetn,
    input wire                  mem_clk,
    input wire                  mem_aresetn,
    
    /* USER INTERFACE */    
    //memory access
    //read cmd
    axis_mem_cmd.slave      s_axis_mem_read_cmd,
    //read status
    axis_mem_status.master  m_axis_mem_read_status,
    //read data stream
    axi_stream.master       m_axis_mem_read_data,
    
    //write cmd
    axis_mem_cmd.slave      s_axis_mem_write_cmd,
    //write status
    axis_mem_status.master  m_axis_mem_write_status,
    //write data stream
    axi_stream.slave        s_axis_mem_write_data,

    // axi_mm.master           network_tx_mem,

    output reg[15:0][31:0]     status_reg

    
    );


axi_mm      network_tx_mem();

/*
 * CLOCK CROSSING
 */

wire        axis_to_dm_mem_write_cmd_tvalid;
wire        axis_to_dm_mem_write_cmd_tready;
wire[71:0]  axis_to_dm_mem_write_cmd_tdata;
assign axis_to_dm_mem_write_cmd_tvalid = s_axis_mem_write_cmd.valid;
assign s_axis_mem_write_cmd.ready = axis_to_dm_mem_write_cmd_tready;
// [71:68] reserved, [67:64] tag, [63:32] address,[31] drr, [30] eof, [29:24] dsa, [23] type, [22:0] btt (bytes to transfer)
assign axis_to_dm_mem_write_cmd_tdata = {8'h0, s_axis_mem_write_cmd.address[31:0], 1'b1, 1'b1, 6'h0, 1'b1, s_axis_mem_write_cmd.length[22:0]};
wire        axis_to_dm_mem_read_cmd_tvalid;
wire        axis_to_dm_mem_read_cmd_tready;
wire[71:0]  axis_to_dm_mem_read_cmd_tdata;
assign axis_to_dm_mem_read_cmd_tvalid = s_axis_mem_read_cmd.valid;
assign s_axis_mem_read_cmd.ready = axis_to_dm_mem_read_cmd_tready;
// [71:68] reserved, [67:64] tag, [63:32] address,[31] drr, [30] eof, [29:24] dsa, [23] type, [22:0] btt (bytes to transfer)
assign axis_to_dm_mem_read_cmd_tdata = {8'h0, s_axis_mem_read_cmd.address[31:0], 1'b1, 1'b1, 6'h0, 1'b1, s_axis_mem_read_cmd.length[22:0]};

wire        axis_mem_cc_to_dm_write_tvalid;
wire        axis_mem_cc_to_dm_write_tready;
wire[511:0] axis_mem_cc_to_dm_write_tdata;
wire[63:0]  axis_mem_cc_to_dm_write_tkeep;
wire        axis_mem_cc_to_dm_write_tlast;

wire        axis_mem_dm_to_cc_read_tvalid;
wire        axis_mem_dm_to_cc_read_tready;
wire[511:0] axis_mem_dm_to_cc_read_tdata;
wire[63:0]  axis_mem_dm_to_cc_read_tkeep;
wire        axis_mem_dm_to_cc_read_tlast;


wire              bram_en_a;
wire [63 : 0]     bram_we_a;
// wire              wea;
wire [19 : 0]     bram_addr_a;
wire [511 : 0]    bram_wrdata_a;
wire [511 : 0]    bram_rddata_a;

wire              bram_en_b;
wire [63 : 0]     bram_we_b;
// wire              web;
wire [19 : 0]     bram_addr_b;
wire [511 : 0]    bram_wrdata_b;
wire [511 : 0]    bram_rddata_b;


    // assign wea      = (bram_we_a != 0);
    // assign web      = (bram_we_b != 0);

axis_data_fifo_512_cc axis_write_data_fifo_mem (
  .s_axis_aclk(user_clk),                // input wire s_axis_aclk
  .s_axis_aresetn(user_aresetn),          // input wire s_axis_aresetn
  .s_axis_tvalid(s_axis_mem_write_data.valid),            // input wire s_axis_tvalid
  .s_axis_tready(s_axis_mem_write_data.ready),            // output wire s_axis_tready
  .s_axis_tdata(s_axis_mem_write_data.data),              // input wire [255 : 0] s_axis_tdata
  .s_axis_tkeep(s_axis_mem_write_data.keep),              // input wire [31 : 0] s_axis_tkeep
  .s_axis_tlast(s_axis_mem_write_data.last),              // input wire s_axis_tlast
   
  .m_axis_aclk(mem_clk),                // input wire m_axis_aclk
  .m_axis_tvalid(axis_mem_cc_to_dm_write_tvalid),            // output wire m_axis_tvalid
  .m_axis_tready(axis_mem_cc_to_dm_write_tready),            // input wire m_axis_tready
  .m_axis_tdata(axis_mem_cc_to_dm_write_tdata),              // output wire [255 : 0] m_axis_tdata
  .m_axis_tkeep(axis_mem_cc_to_dm_write_tkeep),              // output wire [31 : 0] m_axis_tkeep
  .m_axis_tlast(axis_mem_cc_to_dm_write_tlast)              // output wire m_axis_tlast
);

axis_data_fifo_512_cc axis_read_data_fifo_mem (
  .s_axis_aclk(mem_clk),                // input wire s_axis_aclk
  .s_axis_aresetn(mem_aresetn),          // input wire s_axis_aresetn
  .s_axis_tvalid(axis_mem_dm_to_cc_read_tvalid),            // input wire s_axis_tvalid
  .s_axis_tready(axis_mem_dm_to_cc_read_tready),            // output wire s_axis_tready
  .s_axis_tdata(axis_mem_dm_to_cc_read_tdata),              // input wire [255 : 0] s_axis_tdata
  .s_axis_tkeep(axis_mem_dm_to_cc_read_tkeep),              // input wire [31 : 0] s_axis_tkeep
  .s_axis_tlast(axis_mem_dm_to_cc_read_tlast),              // input wire s_axis_tlast
   
  .m_axis_aclk(user_clk),                // input wire m_axis_aclk
  .m_axis_tvalid(m_axis_mem_read_data.valid),            // output wire m_axis_tvalid
  .m_axis_tready(m_axis_mem_read_data.ready),            // input wire m_axis_tready
  .m_axis_tdata(m_axis_mem_read_data.data),              // output wire [255 : 0] m_axis_tdata
  .m_axis_tkeep(m_axis_mem_read_data.keep),              // output wire [31 : 0] m_axis_tkeep
  .m_axis_tlast(m_axis_mem_read_data.last)              // output wire m_axis_tlast
);




/*
 * DATA MOVERS
 */
wire s2mm_error;
wire mm2s_error;

//assign m_axis_mem_write_status.valid = 1;
//assign m_axis_mem_write_status.data = 8'h80;

axi_datamover_mem_unaligned datamover_mem (
    .m_axi_mm2s_aclk(mem_clk),// : IN STD_LOGIC;
    .m_axi_mm2s_aresetn(mem_aresetn), //: IN STD_LOGIC;
    .mm2s_err(mm2s_error), //: OUT STD_LOGIC;
    .m_axis_mm2s_cmdsts_aclk(user_clk), //: IN STD_LOGIC;
    .m_axis_mm2s_cmdsts_aresetn(user_aresetn), //: IN STD_LOGIC;
    .s_axis_mm2s_cmd_tvalid(axis_to_dm_mem_read_cmd_tvalid), //: IN STD_LOGIC;
    .s_axis_mm2s_cmd_tready(axis_to_dm_mem_read_cmd_tready), //: OUT STD_LOGIC;
    .s_axis_mm2s_cmd_tdata(axis_to_dm_mem_read_cmd_tdata), //: IN STD_LOGIC_VECTOR(71 DOWNTO 0);
    .m_axis_mm2s_sts_tvalid(m_axis_mem_read_status.valid), //: OUT STD_LOGIC;
    .m_axis_mm2s_sts_tready(m_axis_mem_read_status.ready), //: IN STD_LOGIC;
    .m_axis_mm2s_sts_tdata(m_axis_mem_read_status.data), //: OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    .m_axis_mm2s_sts_tkeep(), //: OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    .m_axis_mm2s_sts_tlast(), //: OUT STD_LOGIC;
    .m_axi_mm2s_arid    (network_tx_mem.arid), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_mm2s_araddr  (network_tx_mem.araddr), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axi_mm2s_arlen   (network_tx_mem.arlen), //: OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    .m_axi_mm2s_arsize  (network_tx_mem.arsize), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_mm2s_arburst (network_tx_mem.arburst), //: OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_mm2s_arprot  (network_tx_mem.arprot), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_mm2s_arcache (network_tx_mem.arcache), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_mm2s_aruser  (), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_mm2s_arvalid (network_tx_mem.arvalid), //: OUT STD_LOGIC;
    .m_axi_mm2s_arready (network_tx_mem.arready), //: IN STD_LOGIC;
    .m_axi_mm2s_rdata   (network_tx_mem.rdata), //: IN STD_LOGIC_VECTOR(511 DOWNTO 0);
    .m_axi_mm2s_rresp   (network_tx_mem.rresp), //: IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_mm2s_rlast   (network_tx_mem.rlast), //: IN STD_LOGIC;
    .m_axi_mm2s_rvalid  (network_tx_mem.rvalid), //: IN STD_LOGIC;
    .m_axi_mm2s_rready  (network_tx_mem.rready), //: OUT STD_LOGIC;
    .m_axis_mm2s_tdata(axis_mem_dm_to_cc_read_tdata), //: OUT STD_LOGIC_VECTOR(255 DOWNTO 0);
    .m_axis_mm2s_tkeep(axis_mem_dm_to_cc_read_tkeep), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axis_mm2s_tlast(axis_mem_dm_to_cc_read_tlast), //: OUT STD_LOGIC;
    .m_axis_mm2s_tvalid(axis_mem_dm_to_cc_read_tvalid), //: OUT STD_LOGIC;
    .m_axis_mm2s_tready(axis_mem_dm_to_cc_read_tready), //: IN STD_LOGIC;
    .m_axi_s2mm_aclk(mem_clk), //: IN STD_LOGIC;
    .m_axi_s2mm_aresetn(mem_aresetn), //: IN STD_LOGIC;
    .s2mm_err(s2mm_error), //: OUT STD_LOGIC;
    .m_axis_s2mm_cmdsts_awclk(user_clk), //: IN STD_LOGIC;
    .m_axis_s2mm_cmdsts_aresetn(user_aresetn), //: IN STD_LOGIC;
    .s_axis_s2mm_cmd_tvalid(axis_to_dm_mem_write_cmd_tvalid), //: IN STD_LOGIC;
    .s_axis_s2mm_cmd_tready(axis_to_dm_mem_write_cmd_tready), //: OUT STD_LOGIC;
    .s_axis_s2mm_cmd_tdata(axis_to_dm_mem_write_cmd_tdata), //: IN STD_LOGIC_VECTOR(71 DOWNTO 0);
    .m_axis_s2mm_sts_tvalid(m_axis_mem_write_status.valid), //: OUT STD_LOGIC;
    .m_axis_s2mm_sts_tready(m_axis_mem_write_status.ready), //: IN STD_LOGIC;
    .m_axis_s2mm_sts_tdata(m_axis_mem_write_status.data), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axis_s2mm_sts_tkeep(), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axis_s2mm_sts_tlast(), //: OUT STD_LOGIC;
    .m_axi_s2mm_awid    (network_tx_mem.awid), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awaddr  (network_tx_mem.awaddr), //: OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    .m_axi_s2mm_awlen   (network_tx_mem.awlen), //: OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    .m_axi_s2mm_awsize  (network_tx_mem.awsize), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_s2mm_awburst (network_tx_mem.awburst), //: OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_s2mm_awprot  (network_tx_mem.awprot), //: OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    .m_axi_s2mm_awcache (network_tx_mem.awcache), //: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awuser  (),//: OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    .m_axi_s2mm_awvalid (network_tx_mem.awvalid), //: OUT STD_LOGIC;
    .m_axi_s2mm_awready (network_tx_mem.awready), //: IN STD_LOGIC;
    .m_axi_s2mm_wdata   (network_tx_mem.wdata), //: OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
    .m_axi_s2mm_wstrb   (network_tx_mem.wstrb), //: OUT STD_LOGIC_VECTOR(63 DOWNTO 0);
    .m_axi_s2mm_wlast   (network_tx_mem.wlast), //: OUT STD_LOGIC;
    .m_axi_s2mm_wvalid  (network_tx_mem.wvalid), //: OUT STD_LOGIC;
    .m_axi_s2mm_wready  (network_tx_mem.wready), //: IN STD_LOGIC;
    .m_axi_s2mm_bresp   (network_tx_mem.bresp), //: IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    .m_axi_s2mm_bvalid  (network_tx_mem.bvalid), //: IN STD_LOGIC;
    .m_axi_s2mm_bready  (network_tx_mem.bready), //: OUT STD_LOGIC;
    .s_axis_s2mm_tdata(axis_mem_cc_to_dm_write_tdata), //: IN STD_LOGIC_VECTOR(511 DOWNTO 0);
    .s_axis_s2mm_tkeep(axis_mem_cc_to_dm_write_tkeep), //: IN STD_LOGIC_VECTOR(63 DOWNTO 0);
    .s_axis_s2mm_tlast(axis_mem_cc_to_dm_write_tlast), //: IN STD_LOGIC;
    .s_axis_s2mm_tvalid(axis_mem_cc_to_dm_write_tvalid), //: IN STD_LOGIC;
    .s_axis_s2mm_tready(axis_mem_cc_to_dm_write_tready) //: OUT STD_LOGIC;
);

axi_bram_ctrl_512 axi_bram_ctrl_512_inst (
  .s_axi_aclk   (mem_clk),        // input wire s_axi_aclk
  .s_axi_aresetn(mem_aresetn),  // input wire s_axi_aresetn
  .s_axi_awid   (network_tx_mem.awid),        // input wire [0 : 0] s_axi_awid
  .s_axi_awaddr (network_tx_mem.awaddr),    // input wire [18 : 0] s_axi_awaddr
  .s_axi_awlen  (network_tx_mem.awlen),      // input wire [7 : 0] s_axi_awlen
  .s_axi_awsize (network_tx_mem.awsize),    // input wire [2 : 0] s_axi_awsize
  .s_axi_awburst(network_tx_mem.awburst),  // input wire [1 : 0] s_axi_awburst
  .s_axi_awlock (0),    // input wire s_axi_awlock
  .s_axi_awcache(network_tx_mem.awcache),  // input wire [3 : 0] s_axi_awcache
  .s_axi_awprot (network_tx_mem.awprot),    // input wire [2 : 0] s_axi_awprot
  .s_axi_awvalid(network_tx_mem.awvalid),  // input wire s_axi_awvalid
  .s_axi_awready(network_tx_mem.awready),  // output wire s_axi_awready
  .s_axi_wdata  (network_tx_mem.wdata),      // input wire [511 : 0] s_axi_wdata
  .s_axi_wstrb  (network_tx_mem.wstrb),      // input wire [63 : 0] s_axi_wstrb
  .s_axi_wlast  (network_tx_mem.wlast),      // input wire s_axi_wlast
  .s_axi_wvalid (network_tx_mem.wvalid),    // input wire s_axi_wvalid
  .s_axi_wready (network_tx_mem.wready),    // output wire s_axi_wready
  .s_axi_bid    (network_tx_mem.bid),          // output wire [0 : 0] s_axi_bid
  .s_axi_bresp  (network_tx_mem.bresp),      // output wire [1 : 0] s_axi_bresp
  .s_axi_bvalid (network_tx_mem.bvalid),    // output wire s_axi_bvalid
  .s_axi_bready (network_tx_mem.bready),    // input wire s_axi_bready
  .s_axi_arid   (network_tx_mem.arid),        // input wire [0 : 0] s_axi_arid
  .s_axi_araddr (network_tx_mem.araddr),    // input wire [18 : 0] s_axi_araddr
  .s_axi_arlen  (network_tx_mem.arlen),      // input wire [7 : 0] s_axi_arlen
  .s_axi_arsize (network_tx_mem.arsize),    // input wire [2 : 0] s_axi_arsize
  .s_axi_arburst(network_tx_mem.arburst),  // input wire [1 : 0] s_axi_arburst
  .s_axi_arlock (0),    // input wire s_axi_arlock
  .s_axi_arcache(network_tx_mem.arcache),  // input wire [3 : 0] s_axi_arcache
  .s_axi_arprot (network_tx_mem.arprot),    // input wire [2 : 0] s_axi_arprot
  .s_axi_arvalid(network_tx_mem.arvalid),  // input wire s_axi_arvalid
  .s_axi_arready(network_tx_mem.arready),  // output wire s_axi_arready
  .s_axi_rid    (network_tx_mem.rid),          // output wire [0 : 0] s_axi_rid
  .s_axi_rdata  (network_tx_mem.rdata),      // output wire [511 : 0] s_axi_rdata
  .s_axi_rresp  (network_tx_mem.rresp),      // output wire [1 : 0] s_axi_rresp
  .s_axi_rlast  (network_tx_mem.rlast),      // output wire s_axi_rlast
  .s_axi_rvalid (network_tx_mem.rvalid),    // output wire s_axi_rvalid
  .s_axi_rready (network_tx_mem.rready),    // input wire s_axi_rready
  .bram_rst_a(),        // output wire bram_rst_a
  .bram_clk_a(),        // output wire bram_clk_a
  .bram_en_a(bram_en_a),          // output wire bram_en_a
  .bram_we_a(bram_we_a),          // output wire [63 : 0] bram_we_a
  .bram_addr_a(bram_addr_a),      // output wire [18 : 0] bram_addr_a
  .bram_wrdata_a(bram_wrdata_a),  // output wire [511 : 0] bram_wrdata_a
  .bram_rddata_a(bram_rddata_a),  // input wire [511 : 0] bram_rddata_a
  .bram_rst_b(),        // output wire bram_rst_b
  .bram_clk_b(),        // output wire bram_clk_b
  .bram_en_b(bram_en_b),          // output wire bram_en_b
  .bram_we_b(bram_we_b),          // output wire [63 : 0] bram_we_b
  .bram_addr_b(bram_addr_b),      // output wire [18 : 0] bram_addr_b
  .bram_wrdata_b(bram_wrdata_b),  // output wire [511 : 0] bram_wrdata_b
  .bram_rddata_b(bram_rddata_b)  // input wire [511 : 0] bram_rddata_b  
);






   xpm_memory_tdpram #(
      .ADDR_WIDTH_A(14),               // DECIMAL
      .ADDR_WIDTH_B(14),               // DECIMAL
      .AUTO_SLEEP_TIME(0),            // DECIMAL
      .BYTE_WRITE_WIDTH_A(8),        // DECIMAL
      .BYTE_WRITE_WIDTH_B(8),        // DECIMAL
      .CASCADE_HEIGHT(0),             // DECIMAL
      .CLOCKING_MODE("common_clock"), // String
      .ECC_MODE("no_ecc"),            // String
      .MEMORY_INIT_FILE("none"),      // String
      .MEMORY_INIT_PARAM("0"),        // String
      .MEMORY_OPTIMIZATION("true"),   // String
      .MEMORY_PRIMITIVE("ultra"),      // String
      .MEMORY_SIZE(8388608),             // DECIMAL
      .MESSAGE_CONTROL(0),            // DECIMAL
      .READ_DATA_WIDTH_A(512),         // DECIMAL
      .READ_DATA_WIDTH_B(512),         // DECIMAL
      .READ_LATENCY_A(2),             // DECIMAL
      .READ_LATENCY_B(2),             // DECIMAL
      .READ_RESET_VALUE_A("0"),       // String
      .READ_RESET_VALUE_B("0"),       // String
      .RST_MODE_A("SYNC"),            // String
      .RST_MODE_B("SYNC"),            // String
      .SIM_ASSERT_CHK(0),             // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
      .USE_EMBEDDED_CONSTRAINT(0),    // DECIMAL
      .USE_MEM_INIT(1),               // DECIMAL
      .WAKEUP_TIME("disable_sleep"),  // String
      .WRITE_DATA_WIDTH_A(512),        // DECIMAL
      .WRITE_DATA_WIDTH_B(512),        // DECIMAL
      .WRITE_MODE_A("no_change"),     // String
      .WRITE_MODE_B("no_change")      // String
   )
   xpm_memory_tdpram_inst (
      .dbiterra(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                       // on the data output of port A.

      .dbiterrb(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                       // on the data output of port A.

      .douta(bram_rddata_a),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
      .doutb(bram_rddata_b),                   // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
      .sbiterra(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                       // on the data output of port A.

      .sbiterrb(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                       // on the data output of port B.

      .addra(bram_addr_a[19:6]),                   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
      .addrb(bram_addr_b[19:6]),                   // ADDR_WIDTH_B-bit input: Address for port B write and read operations.
      .clka(mem_clk),                     // 1-bit input: Clock signal for port A. Also clocks port B when
                                       // parameter CLOCKING_MODE is "common_clock".

      .clkb(mem_clk),                     // 1-bit input: Clock signal for port B when parameter CLOCKING_MODE is
                                       // "independent_clock". Unused when parameter CLOCKING_MODE is
                                       // "common_clock".

      .dina(bram_wrdata_a),                     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
      .dinb(bram_wrdata_b),                     // WRITE_DATA_WIDTH_B-bit input: Data input for port B write operations.
      .ena(bram_en_a),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
                                       // cycles when read or write operations are initiated. Pipelined
                                       // internally.

      .enb(bram_en_b),                       // 1-bit input: Memory enable signal for port B. Must be high on clock
                                       // cycles when read or write operations are initiated. Pipelined
                                       // internally.

      .injectdbiterra(1'b0), // 1-bit input: Controls double bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .injectdbiterrb(1'b0), // 1-bit input: Controls double bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .injectsbiterra(1'b0), // 1-bit input: Controls single bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .injectsbiterrb(1'b0), // 1-bit input: Controls single bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .regcea(1'b1),                 // 1-bit input: Clock Enable for the last register stage on the output
                                       // data path.

      .regceb(1'b1),                 // 1-bit input: Clock Enable for the last register stage on the output
                                       // data path.

      .rsta(1'b0),                     // 1-bit input: Reset signal for the final port A output register stage.
                                       // Synchronously resets output port douta to the value specified by
                                       // parameter READ_RESET_VALUE_A.

      .rstb(1'b0),                     // 1-bit input: Reset signal for the final port B output register stage.
                                       // Synchronously resets output port doutb to the value specified by
                                       // parameter READ_RESET_VALUE_B.

      .sleep(1'b0),                   // 1-bit input: sleep signal to enable the dynamic power saving feature.
      .wea(bram_we_a),                       // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
                                       // for port A input data port dina. 1 bit wide when word-wide writes are
                                       // used. In byte-wide write configurations, each bit controls the
                                       // writing one byte of dina to address addra. For example, to
                                       // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
                                       // is 32, wea would be 4'b0010.

      .web(bram_we_b)                        // WRITE_DATA_WIDTH_B/BYTE_WRITE_WIDTH_B-bit input: Write enable vector
                                       // for port B input data port dinb. 1 bit wide when word-wide writes are
                                       // used. In byte-wide write configurations, each bit controls the
                                       // writing one byte of dinb to address addrb. For example, to
                                       // synchronously write only bits [15-8] of dinb when WRITE_DATA_WIDTH_B
                                       // is 32, web would be 4'b0010.

   );



/*
 * DDR Statistics
 */ 
logic[31:0] user_write_cmd_counter;
logic[31:0] user_write_sts_counter;
logic[31:0] user_write_sts_error_counter;
logic[31:0] user_read_cmd_counter;
logic[31:0] user_read_sts_counter;
logic[31:0] user_read_sts_error_counter;

logic[31:0] write_cmd_counter;
//logic[47:0] write_cmd_length_counter;
logic[31:0] write_word_counter;
logic[31:0] write_pkg_counter;
logic[47:0] write_length_counter;
logic[31:0] write_sts_counter;
logic[31:0] write_sts_error_counter;

logic[31:0] read_cmd_counter;
// logic[47:0] read_cmd_length_counter;
logic[31:0] read_word_counter;
logic[31:0] read_pkg_counter;
logic[47:0] read_length_counter;
logic[31:0] read_sts_counter;
logic[31:0] read_sts_error_counter;


always @(posedge user_clk)
begin
    if (~user_aresetn) begin
        user_write_cmd_counter <= '0;
        user_write_sts_counter <= '0;
        user_write_sts_error_counter <= '0;
        user_read_cmd_counter <= '0;
        user_read_sts_counter <= '0;
        user_read_sts_error_counter <= '0;
    end
    else begin
        if (axis_to_dm_mem_write_cmd_tvalid && axis_to_dm_mem_write_cmd_tready) begin
            user_write_cmd_counter <= user_write_cmd_counter + 1;
        end
        if (m_axis_mem_write_status.valid && m_axis_mem_write_status.ready) begin
            user_write_sts_counter <= user_write_sts_counter + 1;
            //Check if error occured
            if (m_axis_mem_write_status.data[7] != 1'b1) begin
                user_write_sts_error_counter <= user_write_sts_error_counter;
            end
        end
        if (axis_to_dm_mem_read_cmd_tvalid && axis_to_dm_mem_read_cmd_tready) begin
            user_read_cmd_counter <= user_read_cmd_counter + 1;
        end
        if (m_axis_mem_read_status.valid && m_axis_mem_read_status.ready) begin
            user_read_sts_counter <= user_read_sts_counter + 1;
            //Check if error occured
            if (m_axis_mem_read_status.data[7] != 1'b1) begin
                user_read_sts_error_counter <= user_read_sts_error_counter;
            end
        end

    end
end

always @(posedge mem_clk)
begin
    if (~user_aresetn) begin
        write_word_counter <= '0;
        write_pkg_counter <= '0;
        write_length_counter <= '0;
        read_word_counter <= '0;
        read_pkg_counter <= '0;
        read_length_counter <= '0;
    end
    else begin
        if (axis_mem_cc_to_dm_write_tvalid && axis_mem_cc_to_dm_write_tready) begin
            write_word_counter <= write_word_counter + 1;
            if (axis_mem_cc_to_dm_write_tlast) begin
                write_pkg_counter <= write_pkg_counter + 1;
            end
            //Assumes multiple of 8
            case (axis_mem_cc_to_dm_write_tkeep)
                64'hFF: write_length_counter <= write_length_counter + 8;
                64'hFFFF: write_length_counter <= write_length_counter + 16;
                64'hFFFFFF: write_length_counter <= write_length_counter + 24;
                64'hFFFFFFFF: write_length_counter <= write_length_counter + 32;
                64'hFFFFFFFFFF: write_length_counter <= write_length_counter + 40;
                64'hFFFFFFFFFFFF: write_length_counter <= write_length_counter + 48;
                64'hFFFFFFFFFFFFFF: write_length_counter <= write_length_counter + 56;
                64'hFFFFFFFFFFFFFFFF: write_length_counter <= write_length_counter + 64;
            endcase
        end
        if (axis_mem_dm_to_cc_read_tvalid && axis_mem_dm_to_cc_read_tready) begin
            read_word_counter <= read_word_counter + 1;
            if (axis_mem_dm_to_cc_read_tlast) begin
                read_pkg_counter <= read_pkg_counter + 1;
            end
            //Assumes multiple of 8
            case (axis_mem_dm_to_cc_read_tkeep)
                64'hFF: read_length_counter <= read_length_counter + 8;
                64'hFFFF: read_length_counter <= read_length_counter + 16;
                64'hFFFFFF: read_length_counter <= read_length_counter + 24;
                64'hFFFFFFFF: read_length_counter <= read_length_counter + 32;
                64'hFFFFFFFFFF: read_length_counter <= read_length_counter + 40;
                64'hFFFFFFFFFFFF: read_length_counter <= read_length_counter + 48;
                64'hFFFFFFFFFFFFFF: read_length_counter <= read_length_counter + 56;
                64'hFFFFFFFFFFFFFFFF: read_length_counter <= read_length_counter + 64;
            endcase
        end
    end
end

always@(posedge user_clk)begin
    status_reg[0]               <= user_write_cmd_counter;
    status_reg[1]               <= user_write_sts_counter;
    status_reg[2]               <= user_write_sts_error_counter;
    status_reg[3]               <= user_read_cmd_counter;
    status_reg[4]               <= user_read_sts_counter;
    status_reg[5]               <= user_read_sts_error_counter;
    status_reg[6]               <= write_word_counter;
    status_reg[7]               <= write_pkg_counter;
    status_reg[8]               <= write_length_counter;
    status_reg[9]               <= read_word_counter;
    status_reg[10]              <= read_pkg_counter;
    status_reg[11]              <= read_length_counter;
end


// ila_mem_single your_instance_name (
// 	.clk(user_clk), // input wire clk


// 	.probe0(s_axis_mem_write_cmd.ready), // input wire [0:0]  probe0  
// 	.probe1(s_axis_mem_write_cmd.valid), // input wire [0:0]  probe1 
// 	.probe2(s_axis_mem_write_cmd.address[31:0]), // input wire [31:0]  probe2 
// 	.probe3(m_axis_mem_write_status.valid), // input wire [0:0]  probe3 
// 	.probe4(m_axis_mem_write_status.ready), // input wire [0:0]  probe4 
// 	.probe5(m_axis_mem_write_status.data) // input wire [31:0]  probe5
// );

endmodule

`default_nettype wire
