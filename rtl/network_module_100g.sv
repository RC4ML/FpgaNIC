`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/26/2020 08:19:38 PM
// Design Name: 
// Module Name: network_module_100g
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`timescale 1ns / 1ps
`default_nettype none


module network_module_100g
(
    input wire          dclk,
    input wire          user_clk,  
    output wire         net_clk,
    input wire          sys_reset,
    input wire          aresetn,
    output wire         network_init_done,
    
    input wire          gt_refclk_p,
    input wire          gt_refclk_n,
    
	input  wire [3:0] gt_rxp_in,
	input  wire [3:0] gt_rxn_in,
	output wire [3:0] gt_txp_out,
	output wire [3:0] gt_txn_out,

	output wire         user_rx_reset,
	output wire         user_tx_reset,
	output wire        rx_aligned,
	
	//Axi Stream Interface
	axi_stream.master      m_axis_net_rx,
	axi_stream.slave       s_axis_net_tx
);

reg core_reset_tmp = 1'b0;
reg core_reset = 1'b0;

always @(posedge sys_reset or posedge net_clk) begin 
   if (sys_reset) begin
      core_reset_tmp <= 1'b0;
      core_reset     <= 1'b0;
   end
   else begin
      //Hold core in reset until everything is ready
      core_reset_tmp <= !(sys_reset | user_tx_reset | user_rx_reset);
      core_reset     <= core_reset_tmp;
   end
end
assign network_init_done = core_reset;


/*
 * RX
 */
axi_stream    #(.WIDTH(512))  rx_axis();
axi_stream    #(.WIDTH(512))  rx_reg_axis();
axi_stream    #(.WIDTH(512))  axis_net_rx();

/*
 * TX
 */
axi_stream    #(.WIDTH(512))  tx_axis();
axi_stream    #(.WIDTH(512))  tx_reg_axis();
axi_stream    #(.WIDTH(512))  axis_tx_pkg_to_fifo();
axi_stream    #(.WIDTH(512))  axis_tx_padding_to_fifo();
axi_stream    #(.WIDTH(512))  axis_net_tx();




cmac_axis_wrapper cmac_wrapper_inst
(
    .gt_rxp_in(gt_rxp_in),
    .gt_rxn_in(gt_rxn_in),
    .gt_txp_out(gt_txp_out),
    .gt_txn_out(gt_txn_out),
    .gt_ref_clk_p(gt_refclk_p),
    .gt_ref_clk_n(gt_refclk_n),
    .init_clk(dclk),
    .sys_reset(sys_reset),

    .m_rx_axis(rx_axis),
    .s_tx_axis(tx_axis),

    .rx_aligned(rx_aligned), //Todo REmove/rename
    .usr_tx_clk(net_clk),
    .tx_rst(user_tx_reset),
    .rx_rst(user_rx_reset),
    .gt_rxrecclkout() //not used
);


//RX Clock crossing (same clock)

axis_register_slice_512 rx_reg_crossing (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(~(sys_reset | user_rx_reset)),              // input wire aresetn
  .s_axis_tvalid(rx_axis.valid),  // input wire s_axis_tvalid
  .s_axis_tready(rx_axis.ready),  // output wire s_axis_tready
  .s_axis_tdata(rx_axis.data),    // input wire [511 : 0] s_axis_tdata
  .s_axis_tkeep(rx_axis.keep),    // input wire [63 : 0] s_axis_tkeep
  .s_axis_tlast(rx_axis.last),    // input wire s_axis_tlast
  .m_axis_tvalid(rx_reg_axis.valid),  // output wire m_axis_tvalid
  .m_axis_tready(rx_reg_axis.ready),  // input wire m_axis_tready
  .m_axis_tdata(rx_reg_axis.data),    // output wire [511 : 0] m_axis_tdata
  .m_axis_tkeep(rx_reg_axis.keep),    // output wire [63 : 0] m_axis_tkeep
  .m_axis_tlast(rx_reg_axis.last)    // output wire m_axis_tlast
);

/////////////////////////////////////////////////
//reg ready,valid,last;
//reg [511:0] data;
//reg ready_r,valid_r,last_r;
//reg [511:0] data_r;
//reg ready_rr,valid_rr,last_rr;
//reg [511:0] data_rr;
//reg ready_rrr,valid_rrr,last_rrr;
//reg [511:0] data_rrr;
//reg [63:0] keep_rr,keep_r,keep,keep_rrr;

//always @(posedge user_clk)begin
//    ready   <= m_axis_net_rx.ready;
//    valid   <= m_axis_net_rx.valid;
//    last    <= m_axis_net_rx.last;
//    data    <= m_axis_net_rx.data;
//    keep    <= m_axis_net_rx.keep;
//    ready_r <= ready;
//    valid_r <= valid;
//    last_r  <= last;
//    data_r  <= data;
//    keep_r  <= keep;
//    ready_rr <= ready_r;
//    valid_rr <= valid_r;
//    last_rr  <= last_r;
//    data_rr  <= data_r; 
//    keep_rr <= keep_r;   
//    ready_rrr <= ready_rr;
//    valid_rrr <= valid_rr;
//    last_rrr  <= last_rr;
//    data_rrr  <= data_rr; 
//    keep_rrr <= keep_rr;    
//end

//ila_2 RX (
//	.clk(user_clk), // input wire clk


//	.probe0(ready_rrr), // input wire [0:0]  probe0  
//	.probe1(valid_rrr), // input wire [0:0]  probe1 
//	.probe2(last_rrr), // input wire [0:0]  probe2 
//	.probe3(data_rrr), // input wire [511:0]  probe3
//	.probe4(keep_rrr) // input wire [63:0]  probe4
//);

//ila_2 RX (
//	.clk(user_clk), // input wire clk


//	.probe0(m_axis_net_rx.ready), // input wire [0:0]  probe0  
//	.probe1(m_axis_net_rx.valid), // input wire [0:0]  probe1 
//	.probe2(m_axis_net_rx.last), // input wire [0:0]  probe2 
//	.probe3(m_axis_net_rx.data), // input wire [511:0]  probe3
//	.probe4(m_axis_net_rx.keep) // input wire [63:0]  probe4
//);

//reg ready1,valid1,last1;
//reg [511:0] data1;
//reg ready_r1,valid_r1,last_r1;
//reg [511:0] data_r1;
//reg ready_rr1,valid_rr1,last_rr1;
//reg [511:0] data_rr1;
//reg ready_rrr1,valid_rrr1,last_rrr1;
//reg [511:0] data_rrr1;
//reg [63:0] keep_rr1,keep_r1,keep1,keep_rrr1;

//always @(posedge user_clk)begin
//    ready1   <= s_axis_net_tx.ready;
//    valid1   <= s_axis_net_tx.valid;
//    last1    <= s_axis_net_tx.last;
//    data1    <= s_axis_net_tx.data;
//    keep1    <= s_axis_net_tx.keep;
//    ready_r1 <= ready1;
//    valid_r1 <= valid1;
//    last_r1  <= last1;
//    data_r1  <= data1;
//    keep_r1  <= keep1;
//    ready_rr1 <= ready_r1;
//    valid_rr1 <= valid_r1;
//    last_rr1  <= last_r1;
//    data_rr1  <= data_r1; 
//    keep_rr1 <= keep_r1;   
//    ready_rrr1 <= ready_rr1;
//    valid_rrr1 <= valid_rr1;
//    last_rrr1  <= last_rr1;
//    data_rrr1  <= data_rr1; 
//    keep_rrr1 <= keep_rr1;    
//end


//ila_2 TX (
//	.clk(user_clk), // input wire clk


//	.probe0(ready_rrr1), // input wire [0:0]  probe0  
//	.probe1(valid_rrr1), // input wire [0:0]  probe1 
//	.probe2(last_rrr1), // input wire [0:0]  probe2 
//	.probe3(data_rrr1), // input wire [511:0]  probe3
//	.probe4(keep_rrr1) // input wire [63:0]  probe4
//);


axis_data_fifo_512_cc rx_crossing (
  .s_axis_aresetn(~(sys_reset | user_rx_reset)),
  .s_axis_aclk(net_clk),
  .s_axis_tvalid(rx_reg_axis.valid),
  .s_axis_tready(rx_reg_axis.ready),
  .s_axis_tdata(rx_reg_axis.data),
  .s_axis_tkeep(rx_reg_axis.keep),
  .s_axis_tlast(rx_reg_axis.last),
  .m_axis_aclk(user_clk),
  .m_axis_tvalid(axis_net_rx.valid),
  .m_axis_tready(axis_net_rx.ready),
  .m_axis_tdata(axis_net_rx.data),
  .m_axis_tkeep(axis_net_rx.keep),
  .m_axis_tlast(axis_net_rx.last)
);

axis_register_slice_512 axis_register_slice_512_rx (
  .aclk(user_clk),                    // input wire aclk
  .aresetn(1'b1),              // input wire aresetn
  .s_axis_tvalid(axis_net_rx.valid),  // input wire s_axis_tvalid
  .s_axis_tready(axis_net_rx.ready),  // output wire s_axis_tready
  .s_axis_tdata(axis_net_rx.data),    // input wire [511 : 0] s_axis_tdata
  .s_axis_tkeep(axis_net_rx.keep),    // input wire [63 : 0] s_axis_tkeep
  .s_axis_tlast(axis_net_rx.last),    // input wire s_axis_tlast
  .m_axis_tvalid(m_axis_net_rx.valid),  // output wire m_axis_tvalid
  .m_axis_tready(m_axis_net_rx.ready),  // input wire m_axis_tready
  .m_axis_tdata(m_axis_net_rx.data),    // output wire [511 : 0] m_axis_tdata
  .m_axis_tkeep(m_axis_net_rx.keep),    // output wire [63 : 0] m_axis_tkeep
  .m_axis_tlast(m_axis_net_rx.last)    // output wire m_axis_tlast
);

// TX
// Pad Ethernet frames to at least 64B
// Packet FIFO, makes sure that whole packet is passed in a single burst to the CMAC



axis_register_slice_512 tx_reg_crossing (
  .aclk(net_clk),                    // input wire aclk
  .aresetn(aresetn),              // input wire aresetn
  .s_axis_tvalid(tx_reg_axis.valid),  // input wire s_axis_tvalid
  .s_axis_tready(tx_reg_axis.ready),  // output wire s_axis_tready
  .s_axis_tdata(tx_reg_axis.data),    // input wire [511 : 0] s_axis_tdata
  .s_axis_tkeep(tx_reg_axis.keep),    // input wire [63 : 0] s_axis_tkeep
  .s_axis_tlast(tx_reg_axis.last),    // input wire s_axis_tlast
  .m_axis_tvalid(tx_axis.valid),  // output wire m_axis_tvalid
  .m_axis_tready(tx_axis.ready),  // input wire m_axis_tready
  .m_axis_tdata(tx_axis.data),    // output wire [511 : 0] m_axis_tdata
  .m_axis_tkeep(tx_axis.keep),    // output wire [63 : 0] m_axis_tkeep
  .m_axis_tlast(tx_axis.last)    // output wire m_axis_tlast
);


axis_pkg_fifo_512 axis_pkg_fifo_512 (
  .s_axis_aresetn(~(sys_reset | user_rx_reset)),
  .s_axis_aclk(net_clk),
  .s_axis_tvalid(axis_tx_pkg_to_fifo.valid),
  .s_axis_tready(axis_tx_pkg_to_fifo.ready),
  .s_axis_tdata(axis_tx_pkg_to_fifo.data),
  .s_axis_tkeep(axis_tx_pkg_to_fifo.keep),
  .s_axis_tlast(axis_tx_pkg_to_fifo.last),
  .m_axis_tvalid(tx_reg_axis.valid),
  .m_axis_tready(tx_reg_axis.ready),
  .m_axis_tdata(tx_reg_axis.data),
  .m_axis_tkeep(tx_reg_axis.keep),
  .m_axis_tlast(tx_reg_axis.last)
);


axis_data_fifo_512_cc tx_crossing (
  .s_axis_aresetn(aresetn),
  .s_axis_aclk(user_clk),
  .s_axis_tvalid(axis_tx_padding_to_fifo.valid),
  .s_axis_tready(axis_tx_padding_to_fifo.ready),
  .s_axis_tdata(axis_tx_padding_to_fifo.data),
  .s_axis_tkeep(axis_tx_padding_to_fifo.keep),
  .s_axis_tlast(axis_tx_padding_to_fifo.last),
  .m_axis_aclk(net_clk),
  .m_axis_tvalid(axis_tx_pkg_to_fifo.valid),
  .m_axis_tready(axis_tx_pkg_to_fifo.ready),
  .m_axis_tdata(axis_tx_pkg_to_fifo.data),
  .m_axis_tkeep(axis_tx_pkg_to_fifo.keep),
  .m_axis_tlast(axis_tx_pkg_to_fifo.last)
);





ethernet_frame_padding_512_ip ethernet_frame_padding_inst (
  .m_axis_TVALID(axis_tx_padding_to_fifo.valid),
  .m_axis_TREADY(axis_tx_padding_to_fifo.ready),
  .m_axis_TDATA(axis_tx_padding_to_fifo.data),
  .m_axis_TKEEP(axis_tx_padding_to_fifo.keep),
  .m_axis_TLAST(axis_tx_padding_to_fifo.last),
  .s_axis_TVALID(axis_net_tx.valid),
  .s_axis_TREADY(axis_net_tx.ready),
  .s_axis_TDATA(axis_net_tx.data),
  .s_axis_TKEEP(axis_net_tx.keep),
  .s_axis_TLAST(axis_net_tx.last),
  .ap_clk(user_clk),
  .ap_rst_n(aresetn)
);

axis_register_slice_512 axis_register_slice_512_tx (
  .aclk(user_clk),                    // input wire aclk
  .aresetn(1'b1),              // input wire aresetn
  .s_axis_tvalid(s_axis_net_tx.valid),  // input wire s_axis_tvalid
  .s_axis_tready(s_axis_net_tx.ready),  // output wire s_axis_tready
  .s_axis_tdata(s_axis_net_tx.data),    // input wire [511 : 0] s_axis_tdata
  .s_axis_tkeep(s_axis_net_tx.keep),    // input wire [63 : 0] s_axis_tkeep
  .s_axis_tlast(s_axis_net_tx.last),    // input wire s_axis_tlast
  .m_axis_tvalid(axis_net_tx.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_net_tx.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_net_tx.data),    // output wire [511 : 0] m_axis_tdata
  .m_axis_tkeep(axis_net_tx.keep),    // output wire [63 : 0] m_axis_tkeep
  .m_axis_tlast(axis_net_tx.last)    // output wire m_axis_tlast
);
// ila_0 ila_in (
// 	.clk(net_clk), // input wire clk


// 	.probe0(0), // input wire [0:0]  probe0  
// 	.probe1(0), // input wire [0:0]  probe1 
// 	.probe2(0), // input wire [0:0]  probe2 
// 	.probe3(0), // input wire [0:0]  probe3 
// 	.probe4(axis_tx_padding_to_fifo.ready), // input wire [0:0]  probe4 
// 	.probe5(axis_tx_padding_to_fifo.valid), // input wire [0:0]  probe5 
// 	.probe6(axis_tx_padding_to_fifo.last), // input wire [0:0]  probe6 
// 	.probe7(axis_tx_padding_to_fifo.data), // input wire [63:0]  probe7 
// 	.probe8(tx_axis.ready), // input wire [0:0]  probe8 
// 	.probe9(tx_axis.valid), // input wire [0:0]  probe9 
// 	.probe10(tx_axis.last), // input wire [0:0]  probe10 
// 	.probe11(tx_axis.data) // input wire [63:0]  probe11
// );


endmodule

`default_nettype wire

