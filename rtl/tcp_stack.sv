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

module tcp_stack #(
    parameter TCP_EN = 1,
    parameter WIDTH = 512,
    parameter RX_DDR_BYPASS_EN = 0
)(
    input wire          net_clk,
    input wire          net_aresetn,
    
    // network interface streams
    axi_stream.slave        s_axis_rx_data,
    axi_stream.master       m_axis_tx_data,

    //TCP Interface
    // memory cmd streams
    axis_mem_cmd.master    m_axis_mem_read_cmd,
    axis_mem_cmd.master    m_axis_mem_write_cmd,
    // memory sts streams
    axis_mem_status.slave     s_axis_mem_read_sts,
    axis_mem_status.slave     s_axis_mem_write_sts,
    // memory data streams
    axi_stream.slave    s_axis_mem_read_data,
    axi_stream.master   m_axis_mem_write_data,
    
    
    //Application interface streams
    axis_meta.slave     s_axis_listen_port,
    axis_meta.master    m_axis_listen_port_status,
   
    axis_meta.slave     s_axis_open_connection,
    axis_meta.master    m_axis_open_status,
    axis_meta.slave     s_axis_close_connection,

    axis_meta.master    m_axis_notifications,
    axis_meta.slave     s_axis_read_package,
    
    axis_meta.master    m_axis_rx_metadata,
    axi_stream.master   m_axis_rx_data,
    
    axis_meta.slave     s_axis_tx_metadata,
    axi_stream.slave    s_axis_tx_data,
    axis_meta.master    m_axis_tx_status,


   input wire[31:0]     local_ip_address,
   output logic         session_count_valid,
   output logic[15:0]   session_count_data
 );

localparam ddrPortNetworkRx = 1;
localparam ddrPortNetworkTx = 0;



// Hash Table signals
axis_meta #(.WIDTH(72))     axis_ht_lup_req();
axis_meta #(.WIDTH(88))     axis_ht_lup_rsp();
axis_meta #(.WIDTH(88))     axis_ht_upd_req();
axis_meta #(.WIDTH(88))     axis_ht_upd_rsp();

// Signals for registering
axi_stream #(.WIDTH(WIDTH) )   axis_rxwrite_data();
axi_stream #(.WIDTH(WIDTH) )   axis_rxread_data();
axi_stream #(.WIDTH(WIDTH) )   axis_txwrite_data();
axi_stream #(.WIDTH(WIDTH) )   axis_txread_data();

axis_meta #(.WIDTH(16))     axis_listen_port();
axis_meta #(.WIDTH(8))      axis_listen_port_status();
axis_meta #(.WIDTH(48))     axis_open_connection();
axis_meta #(.WIDTH(24))     axis_open_status();
axis_meta #(.WIDTH(16))     axis_close_connection();

axis_meta #(.WIDTH(88))     axis_notifications();
axis_meta #(.WIDTH(32))     axis_read_package();
axis_meta #(.WIDTH(16))     axis_rx_metadata();
axis_meta #(.WIDTH(32))     axis_tx_metadata();


wire[31:0] rx_buffer_data_count;
reg[15:0] rx_buffer_data_count_reg;
reg[15:0] rx_buffer_data_count_reg2;
axi_stream #(.WIDTH(WIDTH))     axis_rxbuffer2app();
axi_stream #(.WIDTH(WIDTH))     axis_tcp2rxbuffer();

assign s_axis_mem_read_sts.ready = 1'b1;


//hack for now //TODO
wire[71:0] axis_write_cmd_data ;
wire[71:0] axis_read_cmd_data ;

assign m_axis_mem_write_cmd.address = {32'h0000_0000, axis_write_cmd_data[63:32]};
assign m_axis_mem_write_cmd.length = {9'h00, axis_write_cmd_data[22:0]};
assign m_axis_mem_read_cmd.address = {32'h0000_0000, axis_read_cmd_data[63:32]};
assign m_axis_mem_read_cmd.length = {9'h00, axis_read_cmd_data[22:0]};



//TOE Module with RX_DDR_BYPASS disabled


//TOE Module with RX_DDR_BYPASS enabled
toe_ip toe_inst (
// Data output
.m_axis_tcp_data_TVALID(m_axis_tx_data.valid),
.m_axis_tcp_data_TREADY(m_axis_tx_data.ready),
.m_axis_tcp_data_TDATA(m_axis_tx_data.data), // output [63 : 0] AXI_M_Stream_TDATA
.m_axis_tcp_data_TKEEP(m_axis_tx_data.keep),
.m_axis_tcp_data_TLAST(m_axis_tx_data.last),
// Data input
.s_axis_tcp_data_TVALID(s_axis_rx_data.valid),
.s_axis_tcp_data_TREADY(s_axis_rx_data.ready),
.s_axis_tcp_data_TDATA(s_axis_rx_data.data),
.s_axis_tcp_data_TKEEP(s_axis_rx_data.keep),
.s_axis_tcp_data_TLAST(s_axis_rx_data.last),

// rx buffer read path
.s_axis_rxread_data_TVALID(axis_rxbuffer2app.valid),
.s_axis_rxread_data_TREADY(axis_rxbuffer2app.ready),
.s_axis_rxread_data_TDATA(axis_rxbuffer2app.data),
.s_axis_rxread_data_TKEEP(axis_rxbuffer2app.keep),
.s_axis_rxread_data_TLAST(axis_rxbuffer2app.last),
// rx buffer write path
.m_axis_rxwrite_data_TVALID(axis_tcp2rxbuffer.valid),
.m_axis_rxwrite_data_TREADY(axis_tcp2rxbuffer.ready),
.m_axis_rxwrite_data_TDATA(axis_tcp2rxbuffer.data),
.m_axis_rxwrite_data_TKEEP(axis_tcp2rxbuffer.keep),
.m_axis_rxwrite_data_TLAST(axis_tcp2rxbuffer.last),

// tx read commands
.m_axis_txread_cmd_V_TVALID(m_axis_mem_read_cmd.valid),
.m_axis_txread_cmd_V_TREADY(m_axis_mem_read_cmd.ready),
.m_axis_txread_cmd_V_TDATA(axis_read_cmd_data),
//tx write commands
.m_axis_txwrite_cmd_V_TVALID(m_axis_mem_write_cmd.valid),
.m_axis_txwrite_cmd_V_TREADY(m_axis_mem_write_cmd.ready),
.m_axis_txwrite_cmd_V_TDATA(axis_write_cmd_data),
// tx write status
.s_axis_txwrite_sts_V_TVALID(s_axis_mem_write_sts.valid),
.s_axis_txwrite_sts_V_TREADY(s_axis_mem_write_sts.ready),
.s_axis_txwrite_sts_V_TDATA(s_axis_mem_write_sts.data),
// tx read path
.s_axis_txread_data_TVALID(axis_txread_data.valid),
.s_axis_txread_data_TREADY(axis_txread_data.ready),
.s_axis_txread_data_TDATA(axis_txread_data.data),
.s_axis_txread_data_TKEEP(axis_txread_data.keep),
.s_axis_txread_data_TLAST(axis_txread_data.last),
// tx write path
.m_axis_txwrite_data_TVALID(axis_txwrite_data.valid),
.m_axis_txwrite_data_TREADY(axis_txwrite_data.ready),
.m_axis_txwrite_data_TDATA(axis_txwrite_data.data),
.m_axis_txwrite_data_TKEEP(axis_txwrite_data.keep),
.m_axis_txwrite_data_TLAST(axis_txwrite_data.last),
/// SmartCAM I/F ///
.m_axis_session_upd_req_V_TVALID(axis_ht_upd_req.valid),
.m_axis_session_upd_req_V_TREADY(axis_ht_upd_req.ready),
.m_axis_session_upd_req_V_TDATA(axis_ht_upd_req.data),

.s_axis_session_upd_rsp_V_TVALID(axis_ht_upd_rsp.valid),
.s_axis_session_upd_rsp_V_TREADY(axis_ht_upd_rsp.ready),
.s_axis_session_upd_rsp_V_TDATA(axis_ht_upd_rsp.data),

.m_axis_session_lup_req_V_TVALID(axis_ht_lup_req.valid),
.m_axis_session_lup_req_V_TREADY(axis_ht_lup_req.ready),
.m_axis_session_lup_req_V_TDATA(axis_ht_lup_req.data),
.s_axis_session_lup_rsp_V_TVALID(axis_ht_lup_rsp.valid),
.s_axis_session_lup_rsp_V_TREADY(axis_ht_lup_rsp.ready),
.s_axis_session_lup_rsp_V_TDATA(axis_ht_lup_rsp.data),

/* Application Interface */
// listen&close port
.s_axis_listen_port_req_V_V_TVALID(axis_listen_port.valid),
.s_axis_listen_port_req_V_V_TREADY(axis_listen_port.ready),
.s_axis_listen_port_req_V_V_TDATA(axis_listen_port.data),
.m_axis_listen_port_rsp_V_TVALID(axis_listen_port_status.valid),
.m_axis_listen_port_rsp_V_TREADY(axis_listen_port_status.ready),
.m_axis_listen_port_rsp_V_TDATA(axis_listen_port_status.data),

// notification & read request
.m_axis_notification_V_TVALID(axis_notifications.valid),
.m_axis_notification_V_TREADY(axis_notifications.ready),
.m_axis_notification_V_TDATA(axis_notifications.data),
.s_axis_rx_data_req_V_TVALID(axis_read_package.valid),
.s_axis_rx_data_req_V_TREADY(axis_read_package.ready),
.s_axis_rx_data_req_V_TDATA(axis_read_package.data),

// open&close connection
.s_axis_open_conn_req_V_TVALID(axis_open_connection.valid),
.s_axis_open_conn_req_V_TREADY(axis_open_connection.ready),
.s_axis_open_conn_req_V_TDATA(axis_open_connection.data),
.m_axis_open_conn_rsp_V_TVALID(axis_open_status.valid),
.m_axis_open_conn_rsp_V_TREADY(axis_open_status.ready),
.m_axis_open_conn_rsp_V_TDATA(axis_open_status.data),
.s_axis_close_conn_req_V_V_TVALID(axis_close_connection.valid),
.s_axis_close_conn_req_V_V_TREADY(axis_close_connection.ready),
.s_axis_close_conn_req_V_V_TDATA(axis_close_connection.data),

// rx data
.m_axis_rx_data_rsp_metadata_V_V_TVALID(axis_rx_metadata.valid),
.m_axis_rx_data_rsp_metadata_V_V_TREADY(axis_rx_metadata.ready),
.m_axis_rx_data_rsp_metadata_V_V_TDATA(axis_rx_metadata.data),
.m_axis_rx_data_rsp_TVALID(m_axis_rx_data.valid),
.m_axis_rx_data_rsp_TREADY(m_axis_rx_data.ready),
.m_axis_rx_data_rsp_TDATA(m_axis_rx_data.data),
.m_axis_rx_data_rsp_TKEEP(m_axis_rx_data.keep),
.m_axis_rx_data_rsp_TLAST(m_axis_rx_data.last),

// tx data
.s_axis_tx_data_req_metadata_V_TVALID(axis_tx_metadata.valid),
.s_axis_tx_data_req_metadata_V_TREADY(axis_tx_metadata.ready),
.s_axis_tx_data_req_metadata_V_TDATA(axis_tx_metadata.data),
.s_axis_tx_data_req_TVALID(s_axis_tx_data.valid),
.s_axis_tx_data_req_TREADY(s_axis_tx_data.ready),
.s_axis_tx_data_req_TDATA(s_axis_tx_data.data),
.s_axis_tx_data_req_TKEEP(s_axis_tx_data.keep),
.s_axis_tx_data_req_TLAST(s_axis_tx_data.last),
.m_axis_tx_data_rsp_V_TVALID(m_axis_tx_status.valid),
.m_axis_tx_data_rsp_V_TREADY(m_axis_tx_status.ready),
.m_axis_tx_data_rsp_V_TDATA(m_axis_tx_status.data),

.myIpAddress_V(local_ip_address),
.regSessionCount_V(session_count_data),
.regSessionCount_V_ap_vld(session_count_valid),
//for external RX Buffer
.axis_data_count_V(rx_buffer_data_count_reg2),
.axis_max_data_count_V(16'd1024),

.ap_clk(net_clk),                                                        // input aclk
.ap_rst_n(net_aresetn)                                                   // input aresetn
);
 //RX_DDR_BYPASS_EN




//RX BUFFER FIFO
axis_data_fifo_512_d1024 rx_buffer_fifo (
  .s_axis_aresetn(net_aresetn),          // input wire s_axis_aresetn
  .s_axis_aclk(net_clk),                // input wire s_axis_aclk
  .s_axis_tvalid(axis_tcp2rxbuffer.valid),
  .s_axis_tready(axis_tcp2rxbuffer.ready),
  .s_axis_tdata(axis_tcp2rxbuffer.data),
  .s_axis_tkeep(axis_tcp2rxbuffer.keep),
  .s_axis_tlast(axis_tcp2rxbuffer.last),
  .m_axis_tvalid(axis_rxbuffer2app.valid),
  .m_axis_tready(axis_rxbuffer2app.ready),
  .m_axis_tdata(axis_rxbuffer2app.data),
  .m_axis_tkeep(axis_rxbuffer2app.keep),
  .m_axis_tlast(axis_rxbuffer2app.last),
  .axis_wr_data_count(rx_buffer_data_count),
  .axis_rd_data_count()
);



//register data_count
always @(posedge net_clk) begin
    rx_buffer_data_count_reg <= rx_buffer_data_count[15:0];
    rx_buffer_data_count_reg2 <= rx_buffer_data_count_reg;
end //RX_DDR_BYPASS_EN


logic       ht_insert_failure_count_valid;
logic[15:0] ht_insert_failure_count;

hash_table_ip hash_table_inst (
  .ap_clk(net_clk),
  .ap_rst_n(net_aresetn),
  .s_axis_lup_req_V_TVALID(axis_ht_lup_req.valid),
  .s_axis_lup_req_V_TREADY(axis_ht_lup_req.ready),
  .s_axis_lup_req_V_TDATA(axis_ht_lup_req.data),
  .m_axis_lup_rsp_V_TVALID(axis_ht_lup_rsp.valid),
  .m_axis_lup_rsp_V_TREADY(axis_ht_lup_rsp.ready),
  .m_axis_lup_rsp_V_TDATA(axis_ht_lup_rsp.data),
  .s_axis_upd_req_V_TVALID(axis_ht_upd_req.valid),
  .s_axis_upd_req_V_TREADY(axis_ht_upd_req.ready),
  .s_axis_upd_req_V_TDATA(axis_ht_upd_req.data),
  .m_axis_upd_rsp_V_TVALID(axis_ht_upd_rsp.valid),
  .m_axis_upd_rsp_V_TREADY(axis_ht_upd_rsp.ready),
  .m_axis_upd_rsp_V_TDATA(axis_ht_upd_rsp.data),
  .regInsertFailureCount_V_ap_vld(ht_insert_failure_count_valid),
  .regInsertFailureCount_V(ht_insert_failure_count)
);


assign axis_rxread_data.valid = s_axis_mem_read_data.valid;
assign s_axis_mem_read_data.ready = axis_rxread_data.ready;
assign axis_rxread_data.data = s_axis_mem_read_data.data;
assign axis_rxread_data.keep = s_axis_mem_read_data.keep;
assign axis_rxread_data.last = s_axis_mem_read_data.last;

assign m_axis_mem_write_data.valid = axis_rxwrite_data.valid;
assign axis_rxwrite_data.ready = m_axis_mem_write_data.ready;
assign m_axis_mem_write_data.data = axis_rxwrite_data.data;
assign m_axis_mem_write_data.keep = axis_rxwrite_data.keep;
assign m_axis_mem_write_data.last = axis_rxwrite_data.last;

assign axis_txread_data.valid = s_axis_mem_read_data.valid;
assign s_axis_mem_read_data.ready = axis_txread_data.ready;
assign axis_txread_data.data = s_axis_mem_read_data.data;
assign axis_txread_data.keep = s_axis_mem_read_data.keep;
assign axis_txread_data.last = s_axis_mem_read_data.last;
assign m_axis_mem_write_data.valid = axis_txwrite_data.valid;
assign axis_txwrite_data.ready = m_axis_mem_write_data.ready;
assign m_axis_mem_write_data.data = axis_txwrite_data.data;
assign m_axis_mem_write_data.keep = axis_txwrite_data.keep;
assign m_axis_mem_write_data.last = axis_txwrite_data.last;








// Register slices to avoid combinatorial loops created by HLS due to the new axis INTERFACE (enforced since 19.1)

axis_register_slice_16 listen_port_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(s_axis_listen_port.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_listen_port.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_listen_port.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(axis_listen_port.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_listen_port.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_listen_port.data)    // output wire [7 : 0] m_axis_tdata
);

axis_register_slice_8 port_status_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(axis_listen_port_status.valid),  // input wire s_axis_tvalid
  .s_axis_tready(axis_listen_port_status.ready),  // output wire s_axis_tready
  .s_axis_tdata(axis_listen_port_status.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(m_axis_listen_port_status.valid),  // output wire m_axis_tvalid
  .m_axis_tready(m_axis_listen_port_status.ready),  // input wire m_axis_tready
  .m_axis_tdata(m_axis_listen_port_status.data)    // output wire [7 : 0] m_axis_tdata
);

axis_register_slice_48 open_connection_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(s_axis_open_connection.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_open_connection.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_open_connection.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(axis_open_connection.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_open_connection.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_open_connection.data)    // output wire [7 : 0] m_axis_tdata
);

axis_register_slice_24 open_status_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(axis_open_status.valid),  // input wire s_axis_tvalid
  .s_axis_tready(axis_open_status.ready),  // output wire s_axis_tready
  .s_axis_tdata(axis_open_status.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(m_axis_open_status.valid),  // output wire m_axis_tvalid
  .m_axis_tready(m_axis_open_status.ready),  // input wire m_axis_tready
  .m_axis_tdata(m_axis_open_status.data)    // output wire [7 : 0] m_axis_tdata
);

axis_register_slice_16 close_connection_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(s_axis_close_connection.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_close_connection.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_close_connection.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(axis_close_connection.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_close_connection.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_close_connection.data)    // output wire [7 : 0] m_axis_tdata
);

axis_register_slice_88 notification_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(axis_notifications.valid),  // input wire s_axis_tvalid
  .s_axis_tready(axis_notifications.ready),  // output wire s_axis_tready
  .s_axis_tdata(axis_notifications.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(m_axis_notifications.valid),  // output wire m_axis_tvalid
  .m_axis_tready(m_axis_notifications.ready),  // input wire m_axis_tready
  .m_axis_tdata(m_axis_notifications.data)    // output wire [7 : 0] m_axis_tdata
);

axis_register_slice_32 read_package_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(s_axis_read_package.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_read_package.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_read_package.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(axis_read_package.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_read_package.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_read_package.data)    // output wire [7 : 0] m_axis_tdata
);

axis_register_slice_16 axis_rx_metadata_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(axis_rx_metadata.valid),  // input wire s_axis_tvalid
  .s_axis_tready(axis_rx_metadata.ready),  // output wire s_axis_tready
  .s_axis_tdata(axis_rx_metadata.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(m_axis_rx_metadata.valid),  // output wire m_axis_tvalid
  .m_axis_tready(m_axis_rx_metadata.ready),  // input wire m_axis_tready
  .m_axis_tdata(m_axis_rx_metadata.data)    // output wire [7 : 0] m_axis_tdata
);
axis_register_slice_32 axis_tx_metadata_slice (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(s_axis_tx_metadata.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_tx_metadata.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_tx_metadata.data),    // input wire [7 : 0] s_axis_tdata
  .m_axis_tvalid(axis_tx_metadata.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_tx_metadata.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_tx_metadata.data)    // output wire [7 : 0] m_axis_tdata
);

logic[15:0] read_cmd_counter;
logic[15:0] read_pkg_counter;

always @(posedge net_clk) begin
    if (~net_aresetn) begin
        read_cmd_counter <= '0;
        read_pkg_counter <= '0;
    end
    else begin
        if (m_axis_mem_read_cmd.valid && m_axis_mem_read_cmd.ready) begin
            read_cmd_counter <= read_cmd_counter + 1;
        end
        if (s_axis_mem_read_data.valid && s_axis_mem_read_data.ready && s_axis_mem_read_data.last) begin
            read_pkg_counter <= read_pkg_counter + 1;
        end
    end
end


endmodule

`default_nettype wire
