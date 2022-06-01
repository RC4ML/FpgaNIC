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
 
 module tcp_wrapper #(
    parameter       TIME_OUT_CYCLE =    32'hDF84_7580
 )(
     input wire                 clk,
     input wire                 rstn,
     
     
    //netword interface streams
    axis_meta.master            m_axis_listen_port,
    axis_meta.slave             s_axis_listen_port_status,
    
    axis_meta.master            m_axis_open_connection,
    axis_meta.slave             s_axis_open_status,
    axis_meta.master            m_axis_close_connection,
 
    axis_meta.slave             s_axis_notifications,
    axis_meta.master            m_axis_read_package,
    
    axis_meta.slave             s_axis_rx_metadata,
    axi_stream.slave            s_axis_rx_data,
     
    axis_meta.master            m_axis_tx_metadata,
    axi_stream.master           m_axis_tx_data,
    axis_meta.slave             s_axis_tx_status,

    //tcp set conn interface
    axis_meta.master            m_axis_conn_send,     //21
    axis_meta.master            m_axis_ack_to_send,   //ack to tcp send 22
    axis_meta.master            m_axis_ack_to_recv,   //ack to rcv to set buffer id 21
    axis_meta.slave             s_axis_conn_recv,     //22

    //app interface streams
    axis_meta.slave             s_axis_tx_metadata,
    axi_stream.slave            s_axis_tx_data,
    axis_meta.master            m_axis_tx_status,    

    axis_meta.master            m_axis_rx_metadata,
    axi_stream.master           m_axis_rx_data,

    ///
    input wire[13:0][31:0]		control_reg,
	output wire[15:0][31:0]		status_reg

  );
 

  tcp_control#(
    .TIME_OUT_CYCLE 			(32'd250000000)
)tcp_control_inst(
    .clk						(clk),
    .rst						(~rstn),

    .control_reg				(control_reg),
    .status_reg					(status_reg[7:0]),         

    //tcp send interface
    .m_axis_conn_send           (m_axis_conn_send),
    .m_axis_ack_to_send         (m_axis_ack_to_send),     //ack to tcp send
    .m_axis_ack_to_recv         (m_axis_ack_to_recv),     //ack to rcv to set buffer id
    .s_axis_conn_recv           (s_axis_conn_recv),
    //Application interface streams
    .m_axis_listen_port			(m_axis_listen_port),
    .s_axis_listen_port_status	(s_axis_listen_port_status),
   
    .m_axis_open_connection		(m_axis_open_connection),
    .s_axis_open_status			(s_axis_open_status),
    .m_axis_close_connection	(m_axis_close_connection) 

);



// tcp_control_off_path inst_tcp_control_off_path
// (
//     .clk						(clk),
//     .rst						(~rstn),

//     .control_reg				(control_reg),
//     .status_reg					(status_reg[7:0]),         

//     //Application interface streams
//     .m_axis_listen_port			(m_axis_listen_port),
//     .s_axis_listen_port_status	(s_axis_listen_port_status),
   
//     .m_axis_open_connection		(m_axis_open_connection),
//     .s_axis_open_status			(s_axis_open_status),
//     .m_axis_close_connection	(m_axis_close_connection) 

// );

axis_register_slice_48 tx_metadata_slice (
  .aclk(clk),                    // input wire aclk
  .aresetn(rstn),              // input wire aresetn
  .s_axis_tvalid(s_axis_tx_metadata.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_tx_metadata.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_tx_metadata.data),    // input wire [95 : 0] s_axis_tdata
  .m_axis_tvalid(m_axis_tx_metadata.valid),  // output wire m_axis_tvalid
  .m_axis_tready(m_axis_tx_metadata.ready),  // input wire m_axis_tready
  .m_axis_tdata(m_axis_tx_metadata.data)    // output wire [95 : 0] m_axis_tdata
);

// assign s_axis_tx_metadata.ready         = m_axis_tx_metadata.ready;
// assign m_axis_tx_metadata.valid         = s_axis_tx_metadata.valid;
// assign m_axis_tx_metadata.data          = s_axis_tx_metadata.data;

register_slice_wrapper register_slice_wrapper_inst(
    .aclk                       (clk),
    .aresetn                    (rstn),
    .s_axis                     (s_axis_tx_data),
    .m_axis                     (m_axis_tx_data)
);

axis_register_slice_64 tx_status_slice (
  .aclk(clk),                    // input wire aclk
  .aresetn(rstn),              // input wire aresetn
  .s_axis_tvalid(s_axis_tx_status.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_tx_status.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_tx_status.data),    // input wire [63 : 0] s_axis_tdata
  .m_axis_tvalid(m_axis_tx_status.valid),  // output wire m_axis_tvalid
  .m_axis_tready(m_axis_tx_status.ready),  // input wire m_axis_tready
  .m_axis_tdata(m_axis_tx_status.data)    // output wire [63 : 0] m_axis_tdata
);

// assign s_axis_tx_status.ready           = m_axis_tx_status.ready;
// assign m_axis_tx_status.valid           = s_axis_tx_status.valid;
// assign m_axis_tx_status.data            = s_axis_tx_status.data;


	//////////////notifications buffer ///////////
wire 									fifo_rxmeta_wr_en;
reg 									fifo_rxmeta_rd_en;			

wire 									fifo_rxmeta_empty;	
wire 									fifo_rxmeta_almostfull;		
wire [87:0]								fifo_rxmeta_rd_data;
wire 									fifo_rxmeta_rd_valid;			

assign s_axis_notifications.ready 		= ~fifo_rxmeta_almostfull;
assign s_axis_rx_metadata.ready			=  1'b1;
assign fifo_rxmeta_wr_en				= s_axis_notifications.ready && s_axis_notifications.valid;


fwft_fifo_88_d512 inst_tcp_notice_fifo (
  .clk(clk),                  // input wire clk
  .rst(~rstn),                  // input wire rst
  .din(s_axis_notifications.data),                  // input wire [87 : 0] din
  .wr_en(fifo_rxmeta_wr_en),              // input wire wr_en
  .rd_en(fifo_rxmeta_rd_en),              // input wire rd_en
  .dout(fifo_rxmeta_rd_data),                // output wire [87 : 0] dout
  .full(),                // output wire full
  .empty(fifo_rxmeta_empty),              // output wire empty
  .valid(fifo_rxmeta_rd_valid),              // output wire valid
  .prog_full(fifo_rxmeta_almostfull)      // output wire prog_full
);

assign m_axis_rx_metadata.data          = fifo_rxmeta_rd_data;
assign m_axis_rx_metadata.valid         = fifo_rxmeta_rd_valid;
assign fifo_rxmeta_rd_en                = m_axis_rx_metadata.ready;


////////////////////////////////////////////////////////

reg [31:0]                          read_pkg_data; //{16b'datalength,16b'sessionid}
reg									read_pkg_valid;
reg [31:0]                          rx_data_count;

always @(posedge clk)begin
    if(~rstn)
        read_pkg_data                  <= 1'b0;
    else if(s_axis_notifications.valid & s_axis_notifications.ready)
        read_pkg_data                  <= s_axis_notifications.data[31:0];
    else 
        read_pkg_data                  <= read_pkg_data;
end

always @(posedge clk)begin
    if(~rstn)
        read_pkg_valid                  <= 1'b0;
    else if(s_axis_notifications.valid & s_axis_notifications.ready)
        read_pkg_valid                  <= 1'b1;
    else if(m_axis_read_package.valid & m_axis_read_package.ready)
        read_pkg_valid                  <= 1'b0;
    else 
        read_pkg_valid                  <= read_pkg_valid;
end

assign m_axis_read_package.valid        = read_pkg_valid;
assign m_axis_read_package.data         = read_pkg_data;



axis_data_fifo_512_d4096 inst_rx_data_fifo (
    .s_axis_aresetn(rstn),          // input wire s_axis_aresetn
    .s_axis_aclk(clk),                // input wire s_axis_aclk
    .s_axis_tvalid(s_axis_rx_data.valid),            // input wire s_axis_tvalid
    .s_axis_tready(s_axis_rx_data.ready),            // output wire s_axis_tready
    .s_axis_tdata(s_axis_rx_data.data),              // input wire [511 : 0] s_axis_tdata
    .s_axis_tkeep(s_axis_rx_data.keep),              // input wire [63 : 0] s_axis_tkeep
    .s_axis_tlast(s_axis_rx_data.last),              // input wire s_axis_tlast
    .m_axis_tvalid(m_axis_rx_data.valid),            // output wire m_axis_tvalid
    .m_axis_tready(m_axis_rx_data.ready),            // input wire m_axis_tready
    .m_axis_tdata(m_axis_rx_data.data),              // output wire [511 : 0] m_axis_tdata
    .m_axis_tkeep(m_axis_rx_data.keep),              // output wire [63 : 0] m_axis_tkeep
    .m_axis_tlast(m_axis_rx_data.last),              // output wire m_axis_tlast
    .axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
    .axis_rd_data_count(rx_data_count)  // output wire [31 : 0] axis_rd_data_count
  );

  
  assign s_axis_rx_metadata.ready       = 1'b1;

//////////////////////////debug/////////////////////////////

    reg [31:0]                          rx_meta_overflow_cnt;
    reg [31:0]                          rx_data_overflow_cnt;

    always@(posedge clk)begin
        if(~rstn)begin
            rx_meta_overflow_cnt        <= 1'b0;
        end
        else if(fifo_rxmeta_almostfull)begin
            rx_meta_overflow_cnt        <= rx_meta_overflow_cnt + 1'b1;
        end
        else begin
            rx_meta_overflow_cnt        <= rx_meta_overflow_cnt;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            rx_data_overflow_cnt        <= 1'b0;
        end
        else if(rx_data_count > 1000)begin
            rx_data_overflow_cnt        <= rx_data_overflow_cnt + 1'b1;
        end
        else begin
            rx_data_overflow_cnt        <= rx_data_overflow_cnt;
        end
    end    

    assign status_reg[8]                = rx_meta_overflow_cnt;
    assign status_reg[9]                = rx_data_overflow_cnt;
    
    
    
ila_mem_single recv_inst (
	.clk(clk), // input wire clk


	.probe0(s_axis_notifications.valid), // input wire [0:0]  probe0  
	.probe1(s_axis_notifications.ready), // input wire [0:0]  probe1 
	.probe2(s_axis_notifications.data[31:0]), // input wire [31:0]  probe2 
	.probe3(m_axis_read_package.valid), // input wire [0:0]  probe3 
	.probe4(m_axis_read_package.ready), // input wire [0:0]  probe4 
	.probe5(s_axis_notifications.data[79:64]) // input wire [31:0]  probe5
);  
    
    

 endmodule
 
 `default_nettype wire

 



 