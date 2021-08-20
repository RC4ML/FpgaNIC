/*
 * Copyright (c) 2020, Systems Group, ETH Zurich
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

module hyperloglog_role #(
    parameter NUM_ROLE_DDR_CHANNELS = 0
) (
    input wire      user_clk,
    input wire      user_aresetn,
    input wire      pcie_clk,
    input wire      pcie_aresetn,
   
    /* CONTROL INTERFACE */
    axi_lite.slave      s_axil,

    /* NETWORK  - TCP/IP INTERFACE */
    // open port for listening
    axis_meta.master    m_axis_listen_port,
    axis_meta.slave     s_axis_listen_port_status,
   
    axis_meta.master    m_axis_open_connection,
    axis_meta.slave     s_axis_open_status,
    axis_meta.master    m_axis_close_connection,

    axis_meta.slave     s_axis_notifications,
    axis_meta.master    m_axis_read_package,
    
    axis_meta.slave     s_axis_rx_metadata,
    axi_stream.slave    s_axis_rx_data,
    
    axis_meta.master    m_axis_tx_metadata,
    axi_stream.master   m_axis_tx_data,
    axis_meta.slave     s_axis_tx_status,
    
    /* NETWORK - UDP/IP INTERFACE */
    axis_meta.slave     s_axis_udp_rx_metadata,
    axi_stream.slave    s_axis_udp_rx_data,
    axis_meta.master    m_axis_udp_tx_metadata,
    axi_stream.master   m_axis_udp_tx_data,


    /* NETWORK - RDMA INTERFACE */
    axis_meta.slave             s_axis_roce_rx_read_cmd,
    axis_meta.slave             s_axis_roce_rx_write_cmd,
    axi_stream.slave            s_axis_roce_rx_write_data,
    axis_meta.master            m_axis_roce_tx_meta,
    axi_stream.master           m_axis_roce_tx_data,

    /* MEMORY INTERFACE */
    // read command
    axis_mem_cmd.master     m_axis_mem_read_cmd[NUM_DDR_CHANNELS],
    // read status
    axis_mem_status.slave   s_axis_mem_read_status[NUM_DDR_CHANNELS],
    // read data stream
    axi_stream.slave        s_axis_mem_read_data[NUM_DDR_CHANNELS],
    
    // write command
    axis_mem_cmd.master     m_axis_mem_write_cmd[NUM_DDR_CHANNELS],
    // write status
    axis_mem_status.slave   s_axis_mem_write_status[NUM_DDR_CHANNELS],
    // write data stream
    axi_stream.master       m_axis_mem_write_data[NUM_DDR_CHANNELS],
    
    /* DMA INTERFACE */
    axis_mem_cmd.master     m_axis_dma_read_cmd,
    axis_mem_cmd.master     m_axis_dma_write_cmd,

    axi_stream.slave        s_axis_dma_read_data,
    axi_stream.master       m_axis_dma_write_data

);

/*
 * CarMEn - HyperLogLog
 */
wire axis_carmen_param_valid;
reg axis_carmen_param_ready;
wire[95:0] axis_carmen_param_data;

// Agg
wire       axis_agg_param_valid;
reg        axis_agg_param_ready;
wire[17:0] axis_agg_param_data;

// axi stream for input tuples
axi_stream #(.WIDTH(320))    m_axis_input_tuple(); 
axi_stream #(.WIDTH(320))    axis_input_tuple();
axi_stream #(.WIDTH(512))    axis_read_data(); 

logic [47:0] s_size_tuples;
logic [63:0] s_base_addr;
logic [63:0] s_base_reset_addr;
logic [48:0] s_tuple_counter;

carmen_tx_ip carmen_tx_inst(
    .m_axis_dma_read_cmd_TVALID(m_axis_dma_read_cmd.valid),
    .m_axis_dma_read_cmd_TREADY(m_axis_dma_read_cmd.ready),
    .m_axis_dma_read_cmd_TDATA ({m_axis_dma_read_cmd.length, m_axis_dma_read_cmd.address}),
    .s_axis_param_TVALID       (axis_carmen_param_valid),
    .s_axis_param_TREADY       (axis_carmen_param_ready),
    .s_axis_param_TDATA        (axis_carmen_param_data),
    .aresetn                   (user_aresetn),
    .aclk                      (user_clk)

);


assign m_axis_dma_write_data.data[511:32] = 0;
assign m_axis_dma_write_data.keep[63:4] = 0;

    
hyperloglog_ip hyperloglog_inst(
    .ap_rst_n   (user_aresetn),  
    .ap_clk     (user_clk),
    .s_axis_input_tuple_TVALID     (axis_input_tuple.valid),
    .s_axis_input_tuple_TREADY     (axis_input_tuple.ready), 
    .s_axis_input_tuple_TDATA      (axis_input_tuple.data),
    .s_axis_input_tuple_TKEEP      (axis_input_tuple.keep),
    .s_axis_input_tuple_TLAST      (axis_input_tuple.last), 
       
    .m_axis_write_cmd_V_TVALID     (m_axis_dma_write_cmd.valid),
    .m_axis_write_cmd_V_TREADY     (m_axis_dma_write_cmd.ready),
    .m_axis_write_cmd_V_TDATA      ({m_axis_dma_write_cmd.length, m_axis_dma_write_cmd.address}),
    
    .m_axis_write_data_TVALID      (m_axis_dma_write_data.valid),
    .m_axis_write_data_TREADY      (m_axis_dma_write_data.ready),
    .m_axis_write_data_TDATA       (m_axis_dma_write_data.data[31:0]),
    .m_axis_write_data_TKEEP       (m_axis_dma_write_data.keep[3:0]),
    .m_axis_write_data_TLAST       (m_axis_dma_write_data.last),
    
    .regBaseAddr_V                 (s_base_addr)

);

axis_512_to_320_converter axis_512_to_320_converter_data(
  .aclk          (user_clk),
  .aresetn       (user_aresetn),
  .m_axis_tready (axis_input_tuple.ready),      
  .m_axis_tvalid (axis_input_tuple.valid),   
  .m_axis_tdata  (axis_input_tuple.data),   
  .m_axis_tkeep  (axis_input_tuple.keep),   
  .m_axis_tlast  (axis_input_tuple.last),     

  .s_axis_tready (s_axis_dma_read_data.ready),
  .s_axis_tvalid (s_axis_dma_read_data.valid),
  .s_axis_tdata  (s_axis_dma_read_data.data),
  .s_axis_tkeep  (s_axis_dma_read_data.keep),
  .s_axis_tlast  (s_axis_dma_read_data.last) 
);



//Signals not used in role
//network
assign m_axis_listen_port.valid        = 1'b0;
assign s_axis_listen_port_status.ready = 1'b1;

assign m_axis_open_connection.valid    = 1'b0;
assign s_axis_open_status.ready        = 1'b1;
assign m_axis_close_connection.valid   = 1'b0;

assign s_axis_notifications.ready      = 1'b1;
assign m_axis_read_package.valid       = 1'b0;

assign s_axis_rx_metadata.ready        = 1'b1;
assign s_axis_rx_data.ready            = 1'b1;

assign m_axis_tx_metadata.valid        = 1'b0;
assign m_axis_tx_data.valid            = 1'b0;
assign s_axis_tx_status.ready          = 1'b1;

assign s_axis_udp_rx_metadata.ready    = 1'b1;
assign s_axis_udp_rx_data.ready        = 1'b1;
assign m_axis_udp_tx_metadata.valid    = 1'b0;
assign m_axis_udp_tx_data.valid        = 1'b0;

//roce
assign s_axis_roce_rx_read_cmd.ready   = 1'b1;
assign s_axis_roce_rx_write_cmd.ready  = 1'b1;
assign s_axis_roce_rx_write_data.ready = 1'b1;

assign m_axis_roce_tx_meta.valid       = 1'b0;
assign m_axis_roce_tx_data.valid       = 1'b0;
assign m_axis_roce_tx_data.keep        = '0;
assign m_axis_roce_tx_data.last        = '0;

//memory
assign m_axis_mem_read_cmd[0].valid = 1'b0;
assign m_axis_mem_read_cmd[0].address = '0;
assign m_axis_mem_read_cmd[0].length = '0;
assign m_axis_mem_write_cmd[0].valid = 1'b0;
assign m_axis_mem_write_cmd[0].address = '0;
assign m_axis_mem_write_cmd[0].length = '0;
assign s_axis_mem_read_data[0].ready = 1'b1;
assign m_axis_mem_write_data[0].valid = 1'b0;
assign m_axis_mem_write_data[0].data = '0;
assign m_axis_mem_write_data[0].keep = '0;
assign m_axis_mem_write_data[0].last = 1'b0;
assign s_axis_mem_read_status[0].ready = 1'b1;
assign s_axis_mem_write_status[0].ready = 1'b1;

assign m_axis_mem_read_cmd[1].valid = 1'b0;
assign m_axis_mem_read_cmd[1].address = '0;
assign m_axis_mem_read_cmd[1].length = '0;
assign m_axis_mem_write_cmd[1].valid = 1'b0;
assign m_axis_mem_write_cmd[1].address = '0;
assign m_axis_mem_write_cmd[1].length = '0;
assign s_axis_mem_read_data[1].ready = 1'b1;
assign m_axis_mem_write_data[1].valid = 1'b0;
assign m_axis_mem_write_data[1].data = '0;
assign m_axis_mem_write_data[1].keep = '0;
assign m_axis_mem_write_data[1].last = 1'b0;
assign s_axis_mem_read_status[1].ready = 1'b1;
assign s_axis_mem_write_status[1].ready = 1'b1;

//dma
//assign m_axis_dma_write_cmd.valid = 1'b0;
//assign m_axis_dma_write_data.valid = 1'b0;
//assign m_axis_dma_write_data.data = '0;
//assign m_axis_dma_write_data.keep = '0;
//assign m_axis_dma_write_data.last = 1'b0;

// Role Controller
hyperloglog_controller controller_inst(
    .pcie_clk                  (pcie_clk),
    .pcie_aresetn              (pcie_aresetn),
    .user_clk                  (user_clk),
    .user_aresetn              (user_aresetn),
    
     // AXI Lite Master Interface connections
    .s_axil                    (s_axil),
    
    // Control streams
    .m_axis_carmen_param_valid (axis_carmen_param_valid),
    .m_axis_carmen_param_ready (axis_carmen_param_ready),
    .m_axis_carmen_param_data  (axis_carmen_param_data),

    .reg_base_addr             (s_base_addr), 
    .reg_reset_base_addr       (s_base_reset_addr),
    .reg_size_tuples           (s_size_tuples), 

    .m_axis_agg_param_valid    (axis_agg_param_valid),  
    .m_axis_agg_param_ready    (axis_agg_param_ready), 
    .m_axis_agg_param_data     (axis_agg_param_data), 
    
    .carmen_tuples_consumed    (tuples_consumed),
    .carmen_tuples_produced    (tuples_produced)
    
);

reg [7:0]    count_input_tuples;
reg [7:0]    count_output_cache_line;

 always @(posedge user_clk) begin
     if (~user_aresetn) begin
         count_input_tuples      <= '0;
         count_output_cache_line <= '0;
     end
     else begin
         if (axis_input_tuple.valid && axis_input_tuple.ready) begin
             count_input_tuples <= count_input_tuples +1;
         end
        
         if (m_axis_dma_write_data.valid && m_axis_dma_write_data.ready) begin
             count_output_cache_line <= count_output_cache_line +1;
         end
 
        
 
     end
 end



reg[31:0] tuples_consumed;
reg[31:0] tuples_produced;
// Statistics
always @(posedge user_clk) begin
    if (~user_aresetn) begin
        tuples_consumed <= '0;
        tuples_produced <= '0;
    end
    else begin
        if (s_axis_dma_read_data.valid && s_axis_dma_read_data.ready) begin
            case (s_axis_dma_read_data.keep)
                64'hFF: tuples_consumed <= tuples_consumed + 1;
                64'hFFFF: tuples_consumed <= tuples_consumed + 2;
                64'hFFFFFF: tuples_consumed <= tuples_consumed + 3;
                64'hFFFFFFFF: tuples_consumed <= tuples_consumed + 4;
                64'hFFFFFFFFFF: tuples_consumed <= tuples_consumed + 5;
                64'hFFFFFFFFFFFF: tuples_consumed <= tuples_consumed + 6;
                64'hFFFFFFFFFFFFFF: tuples_consumed <= tuples_consumed + 7;
                64'hFFFFFFFFFFFFFFFF: tuples_consumed <= tuples_consumed + 8;
            endcase
        end

        if (m_axis_roce_tx_data.valid && m_axis_roce_tx_data.ready) begin
            case (m_axis_roce_tx_data.keep)
                64'hFF: tuples_produced <= tuples_produced + 1;
                64'hFFFF: tuples_produced <= tuples_produced + 2;
                64'hFFFFFF: tuples_produced <= tuples_produced + 3;
                64'hFFFFFFFF: tuples_produced <= tuples_produced + 4;
                64'hFFFFFFFFFF: tuples_produced <= tuples_produced + 5;
                64'hFFFFFFFFFFFF: tuples_produced <= tuples_produced + 6;
                64'hFFFFFFFFFFFFFF: tuples_produced <= tuples_produced + 7;
                64'hFFFFFFFFFFFFFFFF: tuples_produced <= tuples_produced + 8;
            endcase
        end

    end
end

endmodule
`default_nettype wire
