`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/26/2020 08:42:14 PM
// Design Name: 
// Module Name: tcpip_top_example
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

`include "example_module.vh"

module tcpip_top_example
(

    input  wire [1:0][3:0] gt_rxp_in,
    input  wire [1:0][3:0] gt_rxn_in,
    output wire [1:0][3:0] gt_txp_out,
    output wire [1:0][3:0] gt_txn_out,

//    input wire          sys_reset_n,
    input wire [1:0]  gt_refclk_p,
    input wire [1:0]  gt_refclk_n,
    // input wire          dclk_p,
    // input wire          dclk_n,

    //156.25MHz user clock
     input wire     		sys_100M_p,
	 input wire				sys_100M_n 
    

);



/*
 * Clock & Reset Signals
 */
//wire sys_reset;
// User logic clock & reset
wire user_clk;
wire user_aresetn;
/*
 * Clock & Reset Signals
 */
 wire sys_reset;
 wire sys_100M;
 wire sys_clk_100M;
  // Network user clock & reset
wire [1:0] net_clk;
wire [1:0] net_aresetn;

/*
 * Clock Generation
 */
wire uclk; 
wire dclk;

 mmcm_clk #(
     //clk_out_freq = clk_in_freq * MMCM_CLKFBOUT_MULT_F / (MMCM_DIVCLK_DIVIDE * MMCM_CLKOUT0_DIVIDE_F)
     .MMCM_DIVCLK_DIVIDE            (2),
     .MMCM_CLKFBOUT_MULT_F          (20),
     .MMCM_CLKOUT0_DIVIDE_F         (4),
     .MMCM_CLKOUT1_DIVIDE_F         (10),
     .MMCM_CLKOUT2_DIVIDE_F         (2),
     .MMCM_CLKOUT3_DIVIDE_F         (2),
     .MMCM_CLKOUT4_DIVIDE_F         (2),
     .MMCM_CLKOUT5_DIVIDE_F         (2),
     .MMCM_CLKOUT6_DIVIDE_F         (2),
     .MMCM_CLKIN1_PERIOD            (10.000)
 ) user_clk_inst(
     .clk_in_p                   (sys_100M_p),
     .clk_in_n                   (sys_100M_n),
     .rst_in                     (0),
     //////////////////////clkout////////////////////////////
     .mmcm_lock                  (user_aresetn),                  
     .clk_out0                   (user_clk),           
     .clk_out1                   (dclk),              
     .clk_out2                   (), 
     .clk_out3                   (),
     .clk_out4                   (),
     .clk_out5                   (), 
     .clk_out6                   ()       
 );


/*
 * Network Signals
 */

// TCP/IP
    axis_meta #(.WIDTH(16))     axis_tcp_listen_port();
    axis_meta #(.WIDTH(8))      axis_tcp_port_status();
    axis_meta #(.WIDTH(48))     axis_tcp_open_connection();
    axis_meta #(.WIDTH(24))     axis_tcp_open_status();
    axis_meta #(.WIDTH(16))     axis_tcp_close_connection();
    axis_meta #(.WIDTH(88))     axis_tcp_notification();
    axis_meta #(.WIDTH(32))     axis_tcp_read_pkg();
    
    axis_meta #(.WIDTH(16))     axis_tcp_rx_meta();
    axi_stream #(.WIDTH(512))    axis_tcp_rx_data();
    axis_meta #(.WIDTH(48))     axis_tcp_tx_meta();
    axi_stream #(.WIDTH(512))    axis_tcp_tx_data();
    axis_meta #(.WIDTH(64))     axis_tcp_tx_status();
////////////////////////

    axis_meta #(.WIDTH(16))     axis_tcp_listen_port1();
    axis_meta #(.WIDTH(8))      axis_tcp_port_status1();
    axis_meta #(.WIDTH(48))     axis_tcp_open_connection1();
    axis_meta #(.WIDTH(24))     axis_tcp_open_status1();
    axis_meta #(.WIDTH(16))     axis_tcp_close_connection1();
    axis_meta #(.WIDTH(88))     axis_tcp_notification1();
    axis_meta #(.WIDTH(32))     axis_tcp_read_pkg1();
    
    axis_meta #(.WIDTH(16))     axis_tcp_rx_meta1();
    axi_stream #(.WIDTH(512))    axis_tcp_rx_data1();
    axis_meta #(.WIDTH(48))     axis_tcp_tx_meta1();
    axi_stream #(.WIDTH(512))    axis_tcp_tx_data1();
    axis_meta #(.WIDTH(64))     axis_tcp_tx_status1();

reg [31:0]          ipaddr0,ipaddr1;
reg [7:0]           set_board_num0,set_board_num1;

vio_2 set_addr_vio (
  .clk(user_clk),                // input wire clk
  .probe_out0(ipaddr0),  // output wire [31 : 0] probe_out0
  .probe_out1(ipaddr1),  // output wire [31 : 0] probe_out1
  .probe_out2(set_board_num0),  // output wire [7 : 0] probe_out2
  .probe_out3(set_board_num1)  // output wire [7 : 0] probe_out3
);

/*
 * 100G Network Module
 */

network_stack #(
    .WIDTH(512),
    .MAC_ADDRESS (48'hE59D02350A00) // LSB first, 00:0A:35:02:9D:E5
) network_stack_inst (
    /*          gt ports        */
    .gt_rxp_in(gt_rxp_in[0]),
    .gt_rxn_in(gt_rxn_in[0]),
    .gt_txp_out(gt_txp_out[0]),
    .gt_txn_out(gt_txn_out[0]),

//    input wire          sys_reset_n,
    .gt_refclk_p(gt_refclk_p[0]),
    .gt_refclk_n(gt_refclk_n[0]),

    /*          clock           */
    .dclk(dclk),
    .user_clk(user_clk),
    .user_aresetn(user_aresetn),
    .net_clk(net_clk[0]),
    .net_aresetn(net_aresetn[0]),
    // //Control interface
    .set_ip_addr_data(ipaddr0),//32'h0b01d401
    .set_board_number_data(set_board_num0),
   
    //Role interface
    .s_axis_listen_port(axis_tcp_listen_port),
    .m_axis_listen_port_status(axis_tcp_port_status),
    .s_axis_open_connection(axis_tcp_open_connection),
    .m_axis_open_status(axis_tcp_open_status),
    .s_axis_close_connection(axis_tcp_close_connection),
    .m_axis_notifications(axis_tcp_notification),
    .s_axis_read_package(axis_tcp_read_pkg),
    .m_axis_rx_metadata(axis_tcp_rx_meta),
    .m_axis_rx_data(axis_tcp_rx_data),
    .s_axis_tx_metadata(axis_tcp_tx_meta),
    .s_axis_tx_data(axis_tcp_tx_data),
    .m_axis_tx_status(axis_tcp_tx_status)
);


network_stack1 #(
    .WIDTH(512),
    .MAC_ADDRESS (48'hE59D02350A00) // LSB first, 00:0A:35:02:9D:E5
) network_stack_inst1 (
    /*          gt ports        */
    .gt_rxp_in(gt_rxp_in[1]),
    .gt_rxn_in(gt_rxn_in[1]),
    .gt_txp_out(gt_txp_out[1]),
    .gt_txn_out(gt_txn_out[1]),

//    input wire          sys_reset_n,
    .gt_refclk_p(gt_refclk_p[1]),
    .gt_refclk_n(gt_refclk_n[1]),

    /*          clock           */
    .dclk(dclk),
    .user_clk(user_clk),
    .user_aresetn(user_aresetn),
    .net_clk(net_clk[1]),
    .net_aresetn(net_aresetn[1]),
    // //Control interface
    .set_ip_addr_data(ipaddr1),
    .set_board_number_data(set_board_num1),
   
    //Role interface
    .s_axis_listen_port(axis_tcp_listen_port1),
    .m_axis_listen_port_status(axis_tcp_port_status1),
    .s_axis_open_connection(axis_tcp_open_connection1),
    .m_axis_open_status(axis_tcp_open_status1),
    .s_axis_close_connection(axis_tcp_close_connection1),
    .m_axis_notifications(axis_tcp_notification1),
    .s_axis_read_package(axis_tcp_read_pkg1),
    .m_axis_rx_metadata(axis_tcp_rx_meta1),
    .m_axis_rx_data(axis_tcp_rx_data1),
    .s_axis_tx_metadata(axis_tcp_tx_meta1),
    .s_axis_tx_data(axis_tcp_tx_data1),
    .m_axis_tx_status(axis_tcp_tx_status1)
);



////////////////////////debug//////////////////
reg                                 listen_port;
wire    running;
reg [31:0]                          session_id;
reg                                 read_pkg_valid;
reg [31:0]                          rx_cnt;
reg [31:0]		                    error_cnt;
reg [31:0]		                    error_index;
reg [31:0]                          sum_cnt,sum_all_cnt;
reg                                 sum_en,sum_all_en;

reg running_r,running_rr;
always @(posedge user_clk)begin
    running_r <= running;
    running_rr <= running_r;
end

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        listen_port                 <= 1'b0;
    else if(running_r & ~running_rr)
        listen_port                 <= 1'b1;
    else if(axis_tcp_listen_port.valid & axis_tcp_listen_port.ready)
        listen_port                 <= 1'b0;
    else 
        listen_port                 <= listen_port;
end
assign axis_tcp_listen_port.valid = listen_port;
assign axis_tcp_listen_port.data = 16'h1234;
assign axis_tcp_port_status.ready = 1'b1;
assign axis_tcp_notification.ready = 1'b1;

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        session_id                  <= 1'b0;
    else if(axis_tcp_notification.valid & axis_tcp_notification.ready)
        session_id                  <= axis_tcp_notification.data[31:0];
    else 
        session_id                  <= session_id;
end

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        read_pkg_valid              <= 1'b0;
    else if(axis_tcp_notification.valid & axis_tcp_notification.ready)
        read_pkg_valid              <= 1'b1;
    else if(axis_tcp_read_pkg.valid & axis_tcp_read_pkg.ready)
        read_pkg_valid              <= 1'b0;
    else 
        read_pkg_valid              <= read_pkg_valid;
end

assign axis_tcp_read_pkg.valid = read_pkg_valid;
assign axis_tcp_read_pkg.data = session_id;

assign axis_tcp_rx_data.ready = 1'b1;
assign axis_tcp_rx_meta.ready = 1'b1;

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        rx_cnt <= 1'b0;
    else if(axis_tcp_rx_data.ready & axis_tcp_rx_data.valid)
        rx_cnt <= rx_cnt + 1'b1;      
    else
        rx_cnt <= rx_cnt;
end

always @(posedge user_clk)begin
	if(~net_aresetn[0])
		error_cnt <= 1'b0;
	else if((rx_cnt != axis_tcp_rx_data.data[31:0]) && axis_tcp_rx_data.valid && axis_tcp_rx_data.ready)
		error_cnt <= error_cnt + 1'b1;   
	else
		error_cnt <= error_cnt;
end

always @(posedge user_clk)begin
	if(~net_aresetn[0])
		error_index <= 1'b0;
	else if((rx_cnt != axis_tcp_rx_data.data[31:0]) && axis_tcp_rx_data.valid && axis_tcp_rx_data.ready)
		error_index <= rx_cnt;   
	else
		error_index <= error_index;
end


always @(posedge user_clk)begin
    if(~net_aresetn[0])
        sum_en <= 1'b0;
    else if(axis_tcp_tx_meta.valid & axis_tcp_tx_meta.ready)
        sum_en <= 1'b1;
    else if(cnt0 == 32'h1ff_ffff)//(axis_tcp_tx_data.last && tx_flag)
        sum_en <= 1'b0;        
    else
        sum_en <= sum_en;
end

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        sum_cnt <= 1'b0;
    else if(sum_en)
        sum_cnt <= sum_cnt + 1'b1;       
    else
        sum_cnt <= sum_cnt;
end

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        sum_all_en <= 1'b0;
    else if(axis_tcp_rx_meta.valid & axis_tcp_rx_meta.ready)
        sum_all_en <= 1'b1;
    else if(axis_tcp_rx_data.data[31:0] == 32'h1ff_ffff)
        sum_all_en <= 1'b0;        
    else
        sum_all_en <= sum_all_en;
end

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        sum_all_cnt <= 1'b0;
    else if(sum_all_en)
        sum_all_cnt <= sum_all_cnt + 1'b1;       
    else
        sum_all_cnt <= sum_all_cnt;
end

////////////////////////////1/////////////////
reg                                 listen_port1;
reg [31:0]                          session_id11;
reg                                 read_pkg_valid1;
reg [31:0]                          rx_cnt1;

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        listen_port1                 <= 1'b0;
    else if(running_r & ~running_rr)
        listen_port1                 <= 1'b1;
    else if(axis_tcp_listen_port1.valid & axis_tcp_listen_port1.ready)
        listen_port1                 <= 1'b0;
    else 
        listen_port1                 <= listen_port1;
end
assign axis_tcp_listen_port1.valid = listen_port1;
assign axis_tcp_listen_port1.data = 16'h1234;
assign axis_tcp_port_status1.ready = 1'b1;
assign axis_tcp_notification1.ready = 1'b1;

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        session_id11                  <= 1'b0;
    else if(axis_tcp_notification1.valid & axis_tcp_notification1.ready)
        session_id11                  <= axis_tcp_notification1.data[31:0];
    else 
        session_id11                  <= session_id11;
end

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        read_pkg_valid1              <= 1'b0;
    else if(axis_tcp_notification1.valid & axis_tcp_notification1.ready)
        read_pkg_valid1              <= 1'b1;
    else if(axis_tcp_read_pkg1.valid & axis_tcp_read_pkg1.ready)
        read_pkg_valid1              <= 1'b0;
    else 
        read_pkg_valid1              <= read_pkg_valid1;
end

assign axis_tcp_read_pkg1.valid = read_pkg_valid1;
assign axis_tcp_read_pkg1.data = session_id11;

assign axis_tcp_rx_data1.ready = 1'b1;
assign axis_tcp_rx_meta1.ready = 1'b1;

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        rx_cnt1 <= 1'b0;
    else if(axis_tcp_rx_data1.ready & axis_tcp_rx_data1.valid)
        rx_cnt1 <= rx_cnt1 + 1'b1;      
    else
        rx_cnt1 <= rx_cnt1;
end




//////////////////////////////////////////////////////////

// assign axis_tcp_open_connection.valid = 0;
// assign axis_tcp_open_connection.data = 0;
// assign axis_tcp_open_status.ready = 1'b0;


// assign axis_tcp_tx_meta.valid = 0;
// assign axis_tcp_tx_meta.data = 0;

// assign axis_tcp_tx_data.valid = 1'b0;
// assign axis_tcp_tx_data.data = 64'h5678;
// assign axis_tcp_tx_data.keep = 64'hffff_ffff_ffff_ffff;
// assign axis_tcp_tx_data.last = 0;
// assign axis_tcp_tx_status.ready = 0;


vio_0 iperf_tcp_server_vio (
  .clk(user_clk),                // input wire clk
  .probe_out0(running)  // output wire [0 : 0] probe_out0
  //.probe_out1(regTargetIpAddress)  // output wire [31 : 0] probe_out1
);

ila_2 your_instance_name (
	.clk(net_clk), // input wire clk


	.probe0(axis_tcp_rx_data.ready), // input wire [0:0]  probe0  
	.probe1(axis_tcp_rx_data.valid), // input wire [0:0]  probe1 
	.probe2(axis_tcp_rx_data.data), // input wire [63:0]  probe2 
	.probe3(rx_cnt), // input wire [31:0]  probe3 
	.probe4(sum_all_cnt), // input wire [31:0]  probe4 
	.probe5(sum_cnt) // input wire [31:0]  probe5
);


//ila_net probe_ila_net(
//.clk(user_clk),

//.probe0(axis_tcp_close_connection.valid), // input wire [1:0]
//.probe1(axis_tcp_close_connection.ready), // input wire [1:0]
//.probe2(axis_tcp_close_connection.data), // input wire [16:0]
//.probe3(axis_tcp_listen_port.valid), // input wire [1:0]
//.probe4(axis_tcp_listen_port.ready), // input wire [1:0]
//.probe5(axis_tcp_listen_port.data), // input wire [15:0]
//.probe6(axis_tcp_open_connection.valid), // input wire [1:0]
//.probe7(axis_tcp_open_connection.ready), // input wire [1:0]
//.probe8(sum_cnt), // input wire [48:0]
//.probe9(axis_tcp_read_pkg.valid), // input wire [1:0]
//.probe10(axis_tcp_read_pkg.ready), // input wire [1:0]
//.probe11(axis_tcp_read_pkg.data), // input wire [32:0]
//.probe12(axis_tcp_tx_data.valid), // input wire [1:0]
//.probe13(axis_tcp_tx_data.ready), // input wire [1:0]
//.probe14(axis_tcp_tx_data.data), // input wire [64:0]
//.probe15(axis_tcp_tx_meta.valid), // input wire [1:0]
//.probe16(axis_tcp_tx_meta.ready), // input wire [1:0]
//.probe17(axis_tcp_tx_meta.data), // input wire [16:0]
//.probe18(axis_tcp_port_status.valid), // input wire [1:0]
//.probe19(axis_tcp_port_status.ready), // input wire [1:0]
//.probe20(axis_tcp_port_status.data), // input wire [8:0]
//.probe21(axis_tcp_notification.valid), // input wire [1:0]
//.probe22(axis_tcp_notification.ready), // input wire [1:0]
//.probe23(axis_tcp_notification.data), // input wire [88:0]
//.probe24(axis_tcp_open_status.valid), // input wire [1:0]
//.probe25(axis_tcp_open_status.ready), // input wire [1:0]
//.probe26(axis_tcp_open_status.data), // input wire [24:0]
//.probe27(axis_tcp_rx_data.valid), // input wire [1:0]
//.probe28(axis_tcp_rx_data.ready), // input wire [1:0]
//.probe29(axis_tcp_rx_data.data), // input wire [64:0]
//.probe30(axis_tcp_rx_meta.valid), // input wire [1:0]
//.probe31(axis_tcp_rx_meta.ready), // input wire [1:0]
//.probe32(axis_tcp_rx_meta.data), // input wire [16:0]
//.probe33(axis_tcp_tx_status.valid), // input wire [1:0]
//.probe34(axis_tcp_tx_status.ready), // input wire [1:0]
//.probe35(axis_tcp_tx_status.data) // input wire [24:0]
//);

////////////////////0///////////


reg     open_conn_valid0;
reg [1:0][15:0]                          session_id0;
reg                                     flag,tx_flag,tx_flag_r;   
reg                                 tx_meda_valid0,tx_valid0;
reg [31:0]                          cnt_one0,cnt0,cnt_all0,cnt_one_all0;
wire[47:0]                          open_conn_data;

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        open_conn_valid0                 <= 1'b0;
    else if(start_r & ~start_rr)
        open_conn_valid0                 <= 1'b1;
    else if(axis_tcp_open_connection.valid & axis_tcp_open_connection.ready)
        open_conn_valid0                 <= 1'b0;
    else 
        open_conn_valid0                 <= open_conn_valid0;
end

assign axis_tcp_open_connection.valid = open_conn_valid0;
assign axis_tcp_open_connection.data = open_conn_data;
assign axis_tcp_open_status.ready = 1'b1;

always @(posedge user_clk)begin
    if(~net_aresetn[0])begin
        session_id0[flag]                  <= 1'b0;
        flag                                <= 1'b0;
    end
    else if(axis_tcp_open_status.valid & axis_tcp_open_status.ready)begin
        session_id0[flag]                  <= axis_tcp_open_status.data[15:0];
        flag                                <= ~flag;
    end
    else begin
        session_id0[flag]                  <= session_id0[flag];
        flag                                <= flag;
    end
end

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        tx_meda_valid0              <= 1'b0;
    else if((tx_start_r & ~tx_start_rr) || (tx_flag & ~tx_flag_r) || (~tx_flag & tx_flag_r) )
        tx_meda_valid0              <= 1'b1;
    else if(axis_tcp_tx_meta.valid & axis_tcp_tx_meta.ready)
        tx_meda_valid0              <= 1'b0;
    else 
        tx_meda_valid0              <= tx_meda_valid0;
end


always @(posedge user_clk)begin
    if(~net_aresetn[0])
        tx_flag              <= 1'b0;
    else if(axis_tcp_tx_data.last && (cnt0 < cnt_all0))
        tx_flag              <= ~tx_flag;
    else 
        tx_flag              <= tx_flag;
end

always @(posedge user_clk)begin
    tx_flag_r               <= tx_flag;
end


always @(posedge user_clk)begin
    if(~net_aresetn[0])
        tx_valid0              <= 1'b0;
    else if((tx_start_r & ~tx_start_rr) || (tx_flag & ~tx_flag_r) || (~tx_flag & tx_flag_r))
        tx_valid0              <= 1'b1;
    else if(axis_tcp_tx_data.last)
        tx_valid0              <= 1'b0;
    else 
        tx_valid0              <= tx_valid0;
end


always @(posedge user_clk)begin
    if(~net_aresetn[0])
        cnt0 <= 1'b0;
    else if((cnt0 == cnt_all0) & axis_tcp_tx_data.valid & axis_tcp_tx_data.ready)
        cnt0 <= 1'b0;  
    else if(axis_tcp_tx_data.ready & axis_tcp_tx_data.valid)
        cnt0 <= cnt0 + 1'b1;      
    else
        cnt0 <= cnt0;
end

always @(posedge user_clk)begin
    if(~net_aresetn[0])
        cnt_one0 <= 1'b0;
    else if(axis_tcp_tx_data.last)
        cnt_one0 <= 1'b0;  
    else if(axis_tcp_tx_data.ready & axis_tcp_tx_data.valid)
        cnt_one0 <= cnt_one0 + 1'b1;      
    else
        cnt_one0 <= cnt_one0;
end

 always @(posedge user_clk)begin
         cnt_all0 <= (32'h8000_0000 >>> 6) - 1;
         cnt_one_all0 <= 32'h40 - 1;
 end

assign axis_tcp_tx_meta.valid = tx_meda_valid0;
assign axis_tcp_tx_meta.data = {32'h1000,session_id0[tx_flag]};

assign axis_tcp_tx_data.valid = tx_valid0;//1'b1;
assign axis_tcp_tx_data.data = cnt0;
assign axis_tcp_tx_data.keep = 64'hffff_ffff_ffff_ffff;
assign axis_tcp_tx_data.last = (cnt_one0 == cnt_one_all0) & axis_tcp_tx_data.valid & axis_tcp_tx_data.ready;
assign axis_tcp_tx_status.ready = 1;



ila_0 ila0inst (
	.clk(user_clk), // input wire clk


	.probe0(flag), // input wire [0:0]  probe0  
	.probe1(tx_flag), // input wire [0:0]  probe1 
	.probe2(session_id0[0]), // input wire [15:0]  probe2 
	.probe3(session_id0[1]), // input wire [15:0]  probe3 
	.probe4(axis_tcp_tx_data.ready), // input wire [0:0]  probe4 
	.probe5(axis_tcp_tx_data.valid), // input wire [0:0]  probe5 
	.probe6(axis_tcp_tx_data.data) // input wire [47:0]  probe6
);





///1///////////////////////////

reg     open_conn_valid;
wire    start,tx_start;
reg [15:0]                          session_id1;
reg                                 tx_meda_valid,tx_valid;
reg [31:0]                          cnt,cnt_all;

reg start_r,start_rr;
reg tx_start_r,tx_start_rr;
always @(posedge user_clk)begin
    start_r <= start;
    start_rr <= start_r;
end

always @(posedge user_clk)begin
    tx_start_r <= tx_start;
    tx_start_rr <= tx_start_r;
end

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        open_conn_valid                 <= 1'b0;
    else if(start_r & ~start_rr)
        open_conn_valid                 <= 1'b1;
    else if(axis_tcp_open_connection1.valid & axis_tcp_open_connection1.ready)
        open_conn_valid                 <= 1'b0;
    else 
        open_conn_valid                 <= open_conn_valid;
end

assign axis_tcp_open_connection1.valid = open_conn_valid;
assign axis_tcp_open_connection1.data = 48'h1234_0b01_d401;
assign axis_tcp_open_status1.ready = 1'b1;

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        session_id1                  <= 1'b0;
    else if(axis_tcp_open_status1.valid & axis_tcp_open_status1.ready)
        session_id1                  <= axis_tcp_open_status1.data[15:0];
    else 
        session_id1                  <= session_id1;
end

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        tx_meda_valid              <= 1'b0;
    else if(tx_start_r & ~tx_start_rr)
        tx_meda_valid              <= 1'b1;
    else if(axis_tcp_tx_meta1.valid & axis_tcp_tx_meta1.ready)
        tx_meda_valid              <= 1'b0;
    else 
        tx_meda_valid              <= tx_meda_valid;
end

always @(posedge user_clk)begin
    if(~net_aresetn[1])
        tx_valid              <= 1'b0;
    else if(tx_start_r & ~tx_start_rr)
        tx_valid              <= 1'b1;
    else if(axis_tcp_tx_data1.last)
        tx_valid              <= 1'b0;
    else 
        tx_valid              <= tx_valid;
end


always @(posedge user_clk)begin
    if(~net_aresetn[1])
        cnt <= 1'b0;
    else if(axis_tcp_tx_data1.last)
        cnt <= 1'b0;  
    else if(axis_tcp_tx_data1.ready & axis_tcp_tx_data1.valid)
        cnt <= cnt + 1'b1;      
    else
        cnt <= cnt;
end

 always @(posedge user_clk)begin
         cnt_all <= (axis_tcp_tx_meta1.data[47:16]>>>6) - 1;
 end

assign axis_tcp_tx_meta1.valid = tx_meda_valid;
// assign axis_tcp_tx_meta.data = {16'hfd00,session_id};

assign axis_tcp_tx_data1.valid = tx_valid;//1'b1;
assign axis_tcp_tx_data1.data = cnt;
assign axis_tcp_tx_data1.keep = 64'hffff_ffff_ffff_ffff;
assign axis_tcp_tx_data1.last = (cnt == cnt_all) & axis_tcp_tx_data1.valid & axis_tcp_tx_data1.ready;
assign axis_tcp_tx_status1.ready = 1;

///////////////////////////////////////////////////

// assign axis_tcp_listen_port1.valid = 0;
// assign axis_tcp_listen_port1.data = 16'h0;
// assign axis_tcp_port_status1.ready = 1'b0;
// assign axis_tcp_notification1.ready = 1'b0;


// assign axis_tcp_read_pkg1.valid = 1'b0;
// assign axis_tcp_read_pkg1.data = 0;

// assign axis_tcp_rx_data1.ready = 1'b0;
// assign axis_tcp_rx_meta1.ready = 1'b0;

vio_1 iperf_tcp_client_vio (
  .clk(user_clk),                // input wire clk
  .probe_out0(start),  // output wire [0 : 0] probe_out0
  .probe_out1(tx_start),  // output wire [47 : 0] probe_out1
  .probe_out2(open_conn_data),  // output wire [31 : 0] probe_out2
  .probe_out3(axis_tcp_tx_meta1.data)  // output wire [31 : 0] probe_out2
);




//ila_net probe_ila_net1(
// .clk(user_clk),

//.probe0(axis_tcp_close_connection1.valid), // input wire [1:0]
//.probe1(axis_tcp_close_connection1.ready), // input wire [1:0]
//.probe2(axis_tcp_close_connection1.data), // input wire [16:0]
//.probe3(axis_tcp_listen_port1.valid), // input wire [1:0]
//.probe4(axis_tcp_listen_port1.ready), // input wire [1:0]
//.probe5(axis_tcp_listen_port1.data), // input wire [15:0]
//.probe6(axis_tcp_open_connection1.valid), // input wire [1:0]
//.probe7(axis_tcp_open_connection1.ready), // input wire [1:0]
//.probe8(axis_tcp_open_connection1.data), // input wire [48:0]
//.probe9(axis_tcp_read_pkg1.valid), // input wire [1:0]
//.probe10(axis_tcp_read_pkg1.ready), // input wire [1:0]
//.probe11(axis_tcp_read_pkg1.data), // input wire [32:0]
//.probe12(axis_tcp_tx_data1.valid), // input wire [1:0]
//.probe13(axis_tcp_tx_data1.ready), // input wire [1:0]
//.probe14(axis_tcp_tx_data1.data), // input wire [64:0]
//.probe15(axis_tcp_tx_meta1.valid), // input wire [1:0]
//.probe16(axis_tcp_tx_meta1.ready), // input wire [1:0]
//.probe17(axis_tcp_tx_meta1.data), // input wire [16:0]
//.probe18(axis_tcp_port_status1.valid), // input wire [1:0]
//.probe19(axis_tcp_port_status1.ready), // input wire [1:0]
//.probe20(axis_tcp_port_status1.data), // input wire [8:0]
//.probe21(axis_tcp_notification1.valid), // input wire [1:0]
//.probe22(axis_tcp_notification1.ready), // input wire [1:0]
//.probe23(axis_tcp_notification1.data), // input wire [88:0]
//.probe24(axis_tcp_open_status1.valid), // input wire [1:0]
//.probe25(axis_tcp_open_status1.ready), // input wire [1:0]
//.probe26(axis_tcp_open_status1.data), // input wire [24:0]
//.probe27(axis_tcp_rx_data1.valid), // input wire [1:0]
//.probe28(axis_tcp_rx_data1.ready), // input wire [1:0]
//.probe29(axis_tcp_rx_data1.data), // input wire [64:0]
//.probe30(axis_tcp_rx_meta1.valid), // input wire [1:0]
//.probe31(axis_tcp_rx_meta1.ready), // input wire [1:0]
//.probe32(axis_tcp_rx_meta1.data), // input wire [16:0]
//.probe33(axis_tcp_tx_status1.valid), // input wire [1:0]
//.probe34(axis_tcp_tx_status1.ready), // input wire [1:0]
//.probe35(axis_tcp_tx_status1.data) // input wire [24:0]
//);



////////////////////////debug mac//////////////////
//     assign axis_net_rx_data_0[0].ready = 1;
    
//     assign axis_net_tx_data_0[0].keep = 64'hFFFFFFFFFFFFFFFF;
//     assign axis_net_tx_data_0[0].last = (axis_net_tx_data0[7:0] == 8'd127);  
//     assign axis_net_tx_data_0[0].data = {504'b0,axis_net_tx_data0};
//     assign axis_net_tx_data_0[0].valid = axis_net_tx_data_0_valid;  

// always @(posedge net_clk[0])begin
//     start0_r        <= start0;
// end

// always @(posedge net_clk[0])begin
//     if(axis_net_tx_data_0[0].ready & axis_net_tx_data_0[0].valid)
//         axis_net_tx_data0        <= axis_net_tx_data0 + 1'b1;
//     else
//         axis_net_tx_data0        <= axis_net_tx_data0;
// end

// always @(posedge net_clk[0])begin
//     if(~start0_r & start0)
//         axis_net_tx_data_0_valid        <= 1'b1;
//     else if(axis_net_tx_data0[7:0] == 8'd127)
//         axis_net_tx_data_0_valid        <= 1'b0;
//     else
//         axis_net_tx_data_0_valid        <= axis_net_tx_data_0_valid;
// end




//     assign axis_net_rx_data_1[0].ready = 1;
    
//     assign axis_net_tx_data_1[0].keep = 64'hFFFFFFFFFFFFFFFF;
//     assign axis_net_tx_data_1[0].last = (axis_net_tx_data1[7:0] == 8'd127);  
//     assign axis_net_tx_data_1[0].data = {504'b0,axis_net_tx_data1};
//     assign axis_net_tx_data_1[0].valid = axis_net_tx_data_1_valid;  

// always @(posedge net_clk[1])begin
//     start1_r        <= start1;
// end

// always @(posedge net_clk[1])begin
//     if(axis_net_tx_data_1[0].ready & axis_net_tx_data_1[0].valid)
//         axis_net_tx_data1        <= axis_net_tx_data1 + 1'b1;
//     else
//         axis_net_tx_data1        <= axis_net_tx_data1;
// end

// always @(posedge net_clk[1])begin
//     if(~start1_r & start1)
//         axis_net_tx_data_1_valid        <= 1'b1;
//     else if(axis_net_tx_data1[7:0] == 8'd127)
//         axis_net_tx_data_1_valid        <= 1'b0;
//     else
//         axis_net_tx_data_1_valid        <= axis_net_tx_data_1_valid;
// end



// vio_0 vio_0 (
//   .clk(net_clk[0]),                // input wire clk
//   .probe_out0(start0)  // output wire [0 : 0] probe_out0
// );

// vio_0 vio_1 (
//   .clk(net_clk[1]),                // input wire clk
//   .probe_out0(start1)  // output wire [0 : 0] probe_out0
// );


// ila_0 ila0 (
// 	.clk(net_clk[0]), // input wire clk


// 	.probe0(network_init[0]), // input wire [0:0]  probe0  
// 	.probe1(user_rx_reset[0]), // input wire [0:0]  probe1 
// 	.probe2(user_tx_reset[0]), // input wire [0:0]  probe2 
// 	.probe3(sys_reset), // input wire [0:0]  probe3 
// 	.probe4(axis_net_rx_data_0[0].ready), // input wire [0:0]  probe4 
// 	.probe5(axis_net_rx_data_0[0].valid), // input wire [0:0]  probe5 
// 	.probe6(axis_net_rx_data_0[0].last), // input wire [0:0]  probe6 
// 	.probe7(axis_net_rx_data_0[0].data), // input wire [63:0]  probe7 
// 	.probe8(axis_net_tx_data_0[0].ready), // input wire [0:0]  probe8 
// 	.probe9(axis_net_tx_data_0[0].valid), // input wire [0:0]  probe9 
// 	.probe10(axis_net_tx_data_0[0].last), // input wire [0:0]  probe10 
// 	.probe11(axis_net_tx_data_0[0].data) // input wire [63:0]  probe11
// );

// ila_0 ila1 (
// 	.clk(net_clk[1]), // input wire clk


// 	.probe0(network_init[1]), // input wire [0:0]  probe0  
// 	.probe1(user_rx_reset[1]), // input wire [0:0]  probe1 
// 	.probe2(user_tx_reset[1]), // input wire [0:0]  probe2 
// 	.probe3(0), // input wire [0:0]  probe3 
// 	.probe4(axis_net_rx_data_1[0].ready), // input wire [0:0]  probe4 
// 	.probe5(axis_net_rx_data_1[0].valid), // input wire [0:0]  probe5 
// 	.probe6(axis_net_rx_data_1[0].last), // input wire [0:0]  probe6 
// 	.probe7(axis_net_rx_data_1[0].data), // input wire [63:0]  probe7 
// 	.probe8(axis_net_tx_data_1[0].ready), // input wire [0:0]  probe8 
// 	.probe9(axis_net_tx_data_1[0].valid), // input wire [0:0]  probe9 
// 	.probe10(axis_net_tx_data_1[0].last), // input wire [0:0]  probe10 
// 	.probe11(axis_net_tx_data_1[0].data) // input wire [63:0]  probe11
// );

   

endmodule

`default_nettype wire