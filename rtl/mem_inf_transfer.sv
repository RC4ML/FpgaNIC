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

module mem_inf_transfer #(
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

    axi_mm.master           network_tx_mem


    );

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
wire [18 : 0]     bram_addr_a;
wire [511 : 0]    bram_wrdata_a;
wire [511 : 0]    bram_rddata_a;

wire              bram_en_b;
wire [63 : 0]     bram_we_b;
// wire              web;
wire [18 : 0]     bram_addr_b;
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

endmodule

`default_nettype wire
