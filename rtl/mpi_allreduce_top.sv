`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2020/02/20 21:50:13
// Design Name: 
// Module Name: hbm_driver
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
`include"example_module.vh"

module mpi_allreduce_top( 
    output wire[15 : 0] pcie_tx_p,
    output wire[15 : 0] pcie_tx_n,
    input wire[15 : 0]  pcie_rx_p, 
    input wire[15 : 0]  pcie_rx_n,

    input wire				sys_clk_p,
    input wire				sys_clk_n,
	input wire				sys_rst_n,
	
	// ///////////////ethernet 
    input  wire [0:0][3:0] gt_rxp_in,
    input  wire [0:0][3:0] gt_rxn_in,
    output wire [0:0][3:0] gt_txp_out,
    output wire [0:0][3:0] gt_txn_out,

    input wire [0:0]  gt_refclk_p,
    input wire [0:0]  gt_refclk_n,

	/*			DDR0 INTERFACE		*/
/////////ddr0 input clock
    input                       ddr0_sys_100M_p,
    input                       ddr0_sys_100M_n,
///////////ddr0 PHY interface
    output                      c0_ddr4_act_n,
    output [16:0]               c0_ddr4_adr,
    output [1:0]                c0_ddr4_ba,
    output [1:0]                c0_ddr4_bg,
    output [0:0]                c0_ddr4_cke,
    output [0:0]                c0_ddr4_odt,
    output [0:0]                c0_ddr4_cs_n,
    output [0:0]                c0_ddr4_ck_t,
    output [0:0]                c0_ddr4_ck_c,
    output                      c0_ddr4_reset_n,
    output                      c0_ddr4_parity,
    inout  [71:0]               c0_ddr4_dq,
    inout  [17:0]               c0_ddr4_dqs_t,
    inout  [17:0]               c0_ddr4_dqs_c,
	/*			DDR1 INTERFACE		*/
////////// /ddr0 input clock
    input                       ddr1_sys_100M_p,
    input                       ddr1_sys_100M_n,
/////////ddr1 PHY interface
    output                      c1_ddr4_act_n,
    output [16:0]               c1_ddr4_adr,
    output [1:0]                c1_ddr4_ba,
    output [1:0]                c1_ddr4_bg,
    output [0:0]                c1_ddr4_cke,
    output [0:0]                c1_ddr4_odt,
    output [0:0]                c1_ddr4_cs_n,
    output [0:0]                c1_ddr4_ck_t,
    output [0:0]                c1_ddr4_ck_c,
    output                      c1_ddr4_reset_n,
    output                      c1_ddr4_parity,
    inout  [71:0]               c1_ddr4_dq,
    inout  [17:0]               c1_ddr4_dqs_t,
    inout  [17:0]               c1_ddr4_dqs_c,


    //100MHz user clock
    input wire     		sys_100M_p,
    input wire				sys_100M_n,
     
    output wire           d32_port     

    );
    
    assign d32_port = 1'b0;



// DMA Signals
axis_mem_cmd                axis_dma_read_cmd[4]();
axis_mem_cmd                axis_dma_write_cmd[4]();
axi_stream                  axis_dma_read_data[4]();
axi_stream                  axis_dma_write_data[4]();
axis_meta #(.WIDTH(96))     bypass_cmd();
axis_meta #(.WIDTH(80))     axis_tcp_recv_read_cnt();

axis_meta #(.WIDTH(160))     axis_get_data_cmd();
axis_meta #(.WIDTH(160))     axis_put_data_cmd();

axis_meta #(.WIDTH(128))     axis_get_data_form_net_cmd();
axis_meta #(.WIDTH(128))     axis_put_data_to_net_cmd();
axis_meta #(.WIDTH(128))     axis_put_data_to_net();

axis_meta #(.WIDTH(21))     axis_conn_send();
axis_meta #(.WIDTH(22))     axis_ack_to_send();     //ack to tcp send
axis_meta #(.WIDTH(21))     axis_ack_to_recv();     //ack to rcv to set buffer id
axis_meta #(.WIDTH(22))     axis_conn_recv();


wire[511:0][31:0]     fpga_control_reg;
wire[511:0][31:0]     fpga_status_reg; 

wire[31:0][511:0]     bypass_control_reg;
wire[31:0][511:0]     bypass_status_reg;

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

//network mem clk
wire mem_clk;

//hbm input clk

wire hbm_clk_100M;

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
 .MMCM_CLKOUT2_DIVIDE_F         (3),
 .MMCM_CLKOUT3_DIVIDE_F         (10),
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
 .clk_out2                   (mem_clk), 
 .clk_out3                   (hbm_clk_100M),
 .clk_out4                   (),
 .clk_out5                   (), 
 .clk_out6                   ()       
);






//reset

reg 					reset,reset_r;
reg[7:0]				reset_cnt;
reg 					user_rstn_i;
reg 					user_rstn;			

always @(posedge pcie_clk)begin
	reset				<= fpga_control_reg[0][0];
	reset_r				<= reset;
	user_rstn         <= pcie_aresetn & reset_cnt[7];
end

always @(posedge pcie_clk)begin
	if(reset & ~reset_r)begin
		reset_cnt		<= 1'b0;
	end
	else if(reset_cnt[7] == 1'b1)begin
		reset_cnt		<= reset_cnt;
	end
	else begin
		reset_cnt		<= reset_cnt + 1'b1;
	end
end

//assign user_rstn = pcie_aresetn & reset_cnt[7];

//   BUFG BUFG_inst (
//      .O(user_rstn), // 1-bit output: Clock output
//      .I(user_rstn_i)  // 1-bit input: Clock input
//   );

/*
 * DMA Interface
 */

dma_inf dma_interface (
	/*HPY INTERFACE */
	.pcie_tx_p						(pcie_tx_p),    // output wire [15 : 0] pci_exp_txp
	.pcie_tx_n						(pcie_tx_n),    // output wire [15 : 0] pci_exp_txn
	.pcie_rx_p						(pcie_rx_p),    // input wire [15 : 0] pci_exp_rxp
	.pcie_rx_n						(pcie_rx_n),    // input wire [15 : 0] pci_exp_rxn

    .sys_clk_p						(sys_clk_p),
    .sys_clk_n						(sys_clk_n),
    .sys_rst_n						(sys_rst_n), 

    /* USER INTERFACE */
    //pcie clock output
    .pcie_clk						(pcie_clk),
    .pcie_aresetn					(pcie_aresetn),
	 
	//user clock input
    .user_clk						(pcie_clk),
    .user_aresetn					(pcie_aresetn),

    //DMA Commands 
    .s_axis_dma_read_cmd            (axis_dma_read_cmd),
    .s_axis_dma_write_cmd           (axis_dma_write_cmd),
	//DMA Data streams
    .m_axis_dma_read_data           (axis_dma_read_data),
    .s_axis_dma_write_data          (axis_dma_write_data),
 

    // CONTROL INTERFACE 
    // Control interface
    .fpga_control_reg               (fpga_control_reg),
    .fpga_status_reg                (fpga_status_reg),

    .axis_tcp_recv_read_cnt         (axis_tcp_recv_read_cnt),     
    .bypass_cmd                     (bypass_cmd),
            //off path cmd
    .m_axis_get_data_cmd(axis_get_data_cmd),
    .m_axis_put_data_cmd(axis_put_data_cmd),
    
    //one side
    .m_axis_get_data_form_net(axis_get_data_form_net_cmd),
    .m_axis_put_data_to_net(axis_put_data_to_net_cmd)
// `ifdef XDMA_BYPASS		
//     // bypass register
// 	,.bypass_control_reg 			(bypass_control_reg),
// 	.bypass_status_reg  			(bypass_status_reg)
// `endif

);

// genvar i;
// generate
// 	for(i = 0; i < 4; i = i + 1) begin

// dma_data_transfer#(
//     .PAGE_SIZE (2*1024*1024)	,
//     .PAGE_NUM  (109)		,
//     .CTRL_NUM  (1024)		 
// )dma_data_transfer_inst( 

//     //user clock input
//     .clk							(pcie_clk),
//     .rstn							(user_rstn),

//     //DMA Commands
// //    .axis_dma_read_cmd				(axis_dma_read_cmd[i]),
//     .axis_dma_write_cmd				(axis_dma_write_cmd[i]),

//     //DMA Data streams      
//     .axis_dma_write_data			(axis_dma_write_data[i]),
// //    .axis_dma_read_data				(),

//     //control reg
//     .transfer_base_addr				({fpga_control_reg[41+i*8],fpga_control_reg[40+i*8]}),

//     .transfer_start_page			(fpga_control_reg[42+i*8]),                      
//     .transfer_length				(fpga_control_reg[43+i*8]),
//     .transfer_offset				(fpga_control_reg[44+i*8]),
//     .work_page_size					(fpga_control_reg[45+i*8]),
//     .transfer_start					(fpga_control_reg[46][i]),
// 	.gpu_read_count					(fpga_control_reg[47+i*8]),
// 	.gpu_write_count				(fpga_status_reg[60+i])

//     );

// 	end
// endgenerate
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
axi_stream #(.WIDTH(512))   axis_tcp_rx_data();
axis_meta #(.WIDTH(48))     axis_tcp_tx_meta();
axi_stream #(.WIDTH(512))   axis_tcp_tx_data();
axis_meta #(.WIDTH(64))     axis_tcp_tx_status();

axis_meta #(.WIDTH(88))     app_axis_tcp_rx_meta();
axi_stream #(.WIDTH(512))   app_axis_tcp_rx_data();
axis_meta #(.WIDTH(48))     app_axis_tcp_tx_meta();
axi_stream #(.WIDTH(512))   app_axis_tcp_tx_data();
axis_meta #(.WIDTH(64))     app_axis_tcp_tx_status();


// tcp_control#(
//     .TIME_OUT_CYCLE 			(32'd250000000)
// )tcp_control_inst(
//     .clk						(pcie_clk),
//     .rst						(~user_rstn),

//     .control_reg				(fpga_control_reg[143:130]),
//     .status_reg					(fpga_status_reg[143:128]),         

//     //tcp send interface
//     .m_axis_conn_send           (axis_conn_send),
//     .m_axis_ack_to_send         (axis_ack_to_send),     //ack to tcp send
//     .m_axis_ack_to_recv         (axis_ack_to_recv),     //ack to rcv to set buffer id
//     .s_axis_conn_recv           (axis_conn_recv),
//     //Application interface streams
//     .m_axis_listen_port			(axis_tcp_listen_port),
//     .s_axis_listen_port_status	(axis_tcp_port_status),
   
//     .m_axis_open_connection		(axis_tcp_open_connection),
//     .s_axis_open_status			(axis_tcp_open_status),
//     .m_axis_close_connection	(axis_tcp_close_connection) 

// );


tcp_wrapper_offpath #(
    .TIME_OUT_CYCLE 			(32'd250000000)
    )inst_tcp_wrapper(
        .clk						(pcie_clk),
        .rstn						(user_rstn),
        
        
       //netword interface streams
        .m_axis_listen_port			(axis_tcp_listen_port),
        .s_axis_listen_port_status	(axis_tcp_port_status),
       
        .m_axis_open_connection		(axis_tcp_open_connection),
        .s_axis_open_status			(axis_tcp_open_status),
        .m_axis_close_connection	(axis_tcp_close_connection), 
    
        .s_axis_notifications       (axis_tcp_notification),
        .m_axis_read_package        (axis_tcp_read_pkg),
        
        .s_axis_rx_metadata         (axis_tcp_rx_meta),
        .s_axis_rx_data             (axis_tcp_rx_data),
        
       .m_axis_tx_metadata         (axis_tcp_tx_meta),
       .m_axis_tx_data             (axis_tcp_tx_data),
       .s_axis_tx_status           (axis_tcp_tx_status),

       //director set conn interface
       .m_axis_conn_send           (axis_conn_send),
       .m_axis_ack_to_send         (axis_ack_to_send),     //ack to tcp send
       .m_axis_ack_to_recv         (axis_ack_to_recv),     //ack to rcv to set buffer id
       .s_axis_conn_recv           (axis_conn_recv),

       //app interface streams
       .s_axis_tx_metadata          (app_axis_tcp_tx_meta),
       .s_axis_tx_data              (app_axis_tcp_tx_data),
       .m_axis_tx_status            (app_axis_tcp_tx_status),    
   
       .m_axis_rx_metadata          (app_axis_tcp_rx_meta), 
       .m_axis_rx_data              (app_axis_tcp_rx_data),
   
       ///
       .control_reg				    (fpga_control_reg[143:130]),
       .status_reg			        (fpga_status_reg[143:128])
   
     );


    // dma_read_data_to_tcp dma_read_data_inst( 
		
	// 	//user clock input
	// 	.clk                        (pcie_clk),
	// 	.rstn                       (user_rstn),

	// 	//DMA Commands
	// 	.axis_dma_read_cmd          (axis_dma_read_cmd[0]),
	// 	//DMA Data streams      
	// 	.axis_dma_read_data         (axis_dma_read_data[0]),
		
	// 	//tcp send
	// 	.m_axis_tx_metadata         (app_axis_tcp_tx_meta),
	// 	.m_axis_tx_data             (app_axis_tcp_tx_data),
	// 	.s_axis_tx_status           (app_axis_tcp_tx_status),

    //     //control reg
    //     .s_axis_conn_send           (axis_conn_send),
    //     .s_axis_conn_ack            (axis_ack_to_send),

    //     .s_axis_send_read_cnt       (axis_tcp_recv_read_cnt),        
    //     .s_axis_cmd                 (bypass_cmd),
	// 	.control_reg                (fpga_control_reg[175:160]),
	// 	.status_reg                 (fpga_status_reg[151:144])

	// );

    // dma_write_data_from_tcp dma_write_data_inst( 
    
    //     //user clock input
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),
    
    //     //DMA Commands
    //     .axis_dma_write_cmd         (axis_dma_write_cmd[0]),
    
    //     //DMA Data streams      
    //     .axis_dma_write_data        (axis_dma_write_data[0]),
    
    //     //tcp send
    //     // .s_axis_notifications       (axis_tcp_notification),
    //     // .m_axis_read_package        (axis_tcp_read_pkg),
        
    //     .s_axis_rx_metadata         (app_axis_tcp_rx_meta),
    //     .s_axis_rx_data             (app_axis_tcp_rx_data),

    //     //control cmd
    //     .s_axis_set_buffer_id       (axis_ack_to_recv),
    //     .m_axis_conn_ack_recv       (axis_conn_recv),
    //     //control reg
    //     .control_reg                (fpga_control_reg[191:176]),
    //     .status_reg                 (fpga_status_reg[159:152])
    
    // );


/*
* 100G Network Module
*/

////////mac module/////////////////////////

wire                            network_init;
wire                            user_rx_reset,user_tx_reset;  
axi_stream #(.WIDTH(512))       axis_net_rx_data();
axi_stream #(.WIDTH(512))       axis_net_tx_data();
assign net_aresetn              = network_init;


network_module_100g network_module_inst
(
    .dclk (dclk),
    .user_clk(pcie_clk),
    .net_clk(net_clk),
    .sys_reset (~pcie_aresetn),
    .aresetn(net_aresetn),
    .network_init_done(network_init),
    
    .gt_refclk_p(gt_refclk_p[0]),
    .gt_refclk_n(gt_refclk_n[0]),
    
    .gt_rxp_in(gt_rxp_in[0]),
    .gt_rxn_in(gt_rxn_in[0]),
    .gt_txp_out(gt_txp_out[0]),
    .gt_txn_out(gt_txn_out[0]),
    
    .user_rx_reset(user_rx_reset),
    .user_tx_reset(user_tx_reset),
    .rx_aligned(),
    
    //master 0
    .m_axis_net_rx(axis_net_rx_data),
    .s_axis_net_tx(axis_net_tx_data)

);



 network_stack #(
 .WIDTH(512),
 .MAC_ADDRESS (48'hE59D02350A00) // LSB first, 00:0A:35:02:9D:E5
 ) network_stack_inst (
 /*          gt ports        */
 // .gt_rxp_in(gt_rxp_in[0]),
 // .gt_rxn_in(gt_rxn_in[0]),
 // .gt_txp_out(gt_txp_out[0]),
 // .gt_txn_out(gt_txn_out[0]),

 // //    input wire          sys_reset_n,
 // .gt_refclk_p(gt_refclk_p[0]),
 // .gt_refclk_n(gt_refclk_n[0]),
     .axis_net_rx_data(axis_net_rx_data),
     .axis_net_tx_data(axis_net_tx_data),
 /*          clock           */
 // .dclk(dclk),
 .user_clk(pcie_clk),
 .user_aresetn(user_rstn),
 .net_clk(net_clk),
 .net_aresetn(net_aresetn),
 .mem_clk(mem_clk),
 // //Control interface
 .set_ip_addr_data(fpga_control_reg[129]),//32'h0b01d401
 .set_board_number_data(fpga_control_reg[128]),
 .mtu(fpga_control_reg[127]),

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
 .m_axis_tx_status(axis_tcp_tx_status),
//  .network_tx_mem(c0_ddr4_axi),
 .status_reg(fpga_status_reg[175:160])
 );

// ////////////////////one side/////////


// dma_put_data_to_net dma_put_data_to_net( 

//     //user clock input
//     .clk                        (pcie_clk),
//     .rstn                       (user_rstn),

//     //DMA Commands
//     .axis_dma_read_cmd          (axis_dma_read_cmd[0]),
//     //DMA Data streams      
//     .axis_dma_read_data         (axis_dma_read_data[0]),
    
//     //tcp send
//     .m_axis_tx_metadata         (app_axis_tcp_tx_meta),
//     .m_axis_tx_data             (app_axis_tcp_tx_data),
//     .s_axis_tx_status           (app_axis_tcp_tx_status),

//     //control reg
//     .s_axis_put_data_to_net     (axis_put_data_to_net),
//     .s_axis_get_data_cmd        (axis_get_data_form_net_cmd),
//     .s_axis_put_data_cmd        (axis_put_data_to_net_cmd),
//     .recv_done                  (recv_done),
//     .control_reg                (fpga_control_reg[239:224]),
//     .status_reg                 (fpga_status_reg[183:176])

// );


// dma_get_data_from_net dma_get_data_from_net_inst( 
    
//     //user clock input
//     .clk                        (pcie_clk),
//     .rstn                       (user_rstn),

//     //DMA Commands
//     .axis_dma_write_cmd         (axis_dma_write_cmd[0]),

//     //DMA Data streams      
//     .axis_dma_write_data        (axis_dma_write_data[0]),

//     //tcp send        
//     .s_axis_rx_metadata         (app_axis_tcp_rx_meta),
//     .s_axis_rx_data             (app_axis_tcp_rx_data),

//     //control reg
//     .m_axis_put_data_to_net     (axis_put_data_to_net),
//     .recv_done                  (recv_done),
//     .control_reg                (fpga_control_reg[255:240]),
//     .status_reg                 (fpga_status_reg[191:184])

// );


////////////////////////////simple KVS////////////////////////////

//  read_dma_send_value read_dma_send_value( 
    
//     //user clock input
//     .clk                        (pcie_clk),
//     .rstn                       (user_rstn),

//     //DMA Commands
//     .axis_dma_read_cmd          (axis_dma_read_cmd[0]),
//     //DMA Data streams      
//     .axis_dma_read_data         (axis_dma_read_data[0]),
    
//     //tcp send
//     .m_axis_tx_metadata         (app_axis_tcp_tx_meta),
//     .m_axis_tx_data             (app_axis_tcp_tx_data),
//     .s_axis_tx_status           (app_axis_tcp_tx_status),


//     .control_reg                (fpga_control_reg[303:288]),
//     .status_reg                 (fpga_status_reg[295:288])

//     );

//     receive_key_write_dma receive_key_write_dma_inst( 
    
//         //user clock input
//         .clk                        (pcie_clk),
//         .rstn                       (user_rstn),
    
//         //DMA Commands
//         .axis_dma_write_cmd         (axis_dma_write_cmd[0]),
    
//         //DMA Data streams      
//         .axis_dma_write_data        (axis_dma_write_data[0]),
    
//         //tcp send        
//         .s_axis_rx_metadata         (app_axis_tcp_rx_meta),
//         .s_axis_rx_data             (app_axis_tcp_rx_data),
    
//         //control reg
//         .control_reg                (fpga_control_reg[319:304]),
//         .status_reg                 (fpga_status_reg[303:296])
    
//         );



///////////////////// HBM interface

// axi_mm #(.ADDR_WIDTH(33),.DATA_WIDTH(256))hbm_axi[32]();
// wire            hbm_clk;
// wire            hbm_rstn;

// hbm_driver inst_hbm_driver(

//     .sys_clk_100M(hbm_clk_100M),
//     .hbm_axi(hbm_axi),
//     .hbm_clk(hbm_clk),
//     .hbm_rstn(hbm_rstn)
//     );

//     matrix_multiply (
//         .clk        (hbm_clk),
//         .rstn       (user_rstn),
    
//         //DMA Commands
//         .matrix0    (hbm_axi[0]), 
//         .matrix1    (hbm_axi[1]),
    
//         // control reg
//         .control_reg(fpga_control_reg[271:256]),
//         .status_reg (fpga_status_reg[271:256])    
     
//      );


//     axi_mm          axis_mem_read0();
//     axi_mm          axis_mem_read1();
//     axi_mm          axis_mem_write0();
//     axi_mm          axis_mem_write1();

//     axi_clock_convert_warpper  #(
//         .WIDTH(256)
//         )inst_read0_clock_convert(
//         .s_axi_aclk         (pcie_clk),
//         .s_axi_aresetn      (user_rstn),
   
//         .m_axi_aclk         (hbm_clk),
//         .m_axi_aresetn      (hbm_rstn),
   
//         .s_axi              (axis_mem_read0),
//         .m_axi              (hbm_axi[0])
//     );

//     axi_clock_convert_warpper #(
//         .WIDTH(256)
//         ) inst_read1_clock_convert(
//         .s_axi_aclk         (pcie_clk),
//         .s_axi_aresetn      (user_rstn),
   
//         .m_axi_aclk         (hbm_clk),
//         .m_axi_aresetn      (hbm_rstn),
   
//         .s_axi              (axis_mem_read1),
//         .m_axi              (hbm_axi[1])
//     );

//     axi_clock_convert_warpper #(
//         .WIDTH(256)
//         ) inst_write0_clock_convert(
//         .s_axi_aclk         (pcie_clk),
//         .s_axi_aresetn      (user_rstn),
   
//         .m_axi_aclk         (hbm_clk),
//         .m_axi_aresetn      (hbm_rstn),
   
//         .s_axi              (axis_mem_write0),
//         .m_axi              (hbm_axi[2])
//     );

//     axi_clock_convert_warpper #(
//         .WIDTH(256)
//         ) inst_write1_clock_convert(
//         .s_axi_aclk         (pcie_clk),
//         .s_axi_aresetn      (user_rstn),
   
//         .m_axi_aclk         (hbm_clk),
//         .m_axi_aresetn      (hbm_rstn),
   
//         .s_axi              (axis_mem_write1),
//         .m_axi              (hbm_axi[3])
//     );




    // mpi_reduce_control mpi_reduce_control_inst( 
    //     //user clock input
    //     .clk(pcie_clk),
    //     .rstn(user_rstn),
        
    //     //dma memory streams
    //     .m_axis_dma_write_cmd(axis_dma_write_cmd[0]),
    //     .m_axis_dma_write_data(axis_dma_write_data[0]),
    //     .m_axis_dma_read_cmd(axis_dma_read_cmd[0]),
    //     .s_axis_dma_read_data(axis_dma_read_data[0]),    
    
    //     //dma memory streams
    //     .m_axis_mem_read0(axis_mem_read0),
    //     .m_axis_mem_read1(axis_mem_read1),
    //     .m_axis_mem_write0(axis_mem_write0),
    //     .m_axis_mem_write1(axis_mem_write1),     
    
    //     //tcp app interface streams
    //     .app_axis_tcp_tx_meta(app_axis_tcp_tx_meta),
    //     .app_axis_tcp_tx_data(app_axis_tcp_tx_data),
    //     .app_axis_tcp_tx_status(app_axis_tcp_tx_status),    
    
    //     .app_axis_tcp_rx_meta(app_axis_tcp_rx_meta),
    //     .app_axis_tcp_rx_data(app_axis_tcp_rx_data),
    
    //     //control reg
    //     .control_reg(fpga_control_reg[271:256]),
    //     .status_reg(fpga_status_reg[287:256])
    
        
    // );


    // hyperloglog hyperloglog_inst(
    //     .clk(pcie_clk),
    //     .rstn(user_rstn),
       
        
    //     /* DMA INTERFACE */
    //     .m_axis_dma_write_cmd(axis_dma_write_cmd[0]),
    //     .m_axis_dma_write_data(axis_dma_write_data[0]),

    //     .s_axis_read_data(app_axis_tcp_rx_data),
    //    //app interface streams
    //     .m_axis_tx_metadata          (app_axis_tcp_tx_meta),
    //     .m_axis_tx_data              (app_axis_tcp_tx_data),
    //     .s_axis_tx_status            (app_axis_tcp_tx_status),    
    
    //     .s_axis_rx_metadata          (app_axis_tcp_rx_meta), 




    //    //control reg
    //     .control_reg(fpga_control_reg[287:272]),
    //     .status_reg(fpga_status_reg[289:288])     
    
    // );

/*     DDR  USER INTERFACE   */
//DDR0 user interface
axi_mm #(.DATA_WIDTH(512),.ADDR_WIDTH(34))         c0_ddr4_axi();
wire            c0_ddr4_clk;
wire            c0_ddr4_rst;
wire            c0_init_complete;
//// DDR1 user interface
axi_mm #(.DATA_WIDTH(512),.ADDR_WIDTH(34))         c1_ddr4_axi();
wire            c1_ddr4_clk;
wire            c1_ddr4_rst;
wire            c1_init_complete;

    // memory cmd streams
axis_mem_cmd    axis_mem_read_cmd();
axis_mem_cmd    axis_mem_write_cmd();
// memory sts streams
axis_mem_status     axis_mem_read_sts();
axis_mem_status     axis_mem_write_sts();
// memory data streams
axi_stream    axis_mem_read_data();
axi_stream   axis_mem_write_data();

axi_mm #(.DATA_WIDTH(512),.ADDR_WIDTH(34))         axis_mem_read0();
axi_mm #(.DATA_WIDTH(512),.ADDR_WIDTH(34))         axis_mem_write0();

axi_mm #(.DATA_WIDTH(512),.ADDR_WIDTH(34))         c1_ddr4_axi_reg();


ddr0_driver inst_ddr0_driver( 
	/*			DDR0 INTERFACE		*/
/////////ddr0 input clock
    .ddr0_sys_100M_p            (ddr0_sys_100M_p),
    .ddr0_sys_100M_n            (ddr0_sys_100M_n),
///////////ddr0 PHY interface
    .c0_ddr4_act_n              (c0_ddr4_act_n),
    .c0_ddr4_adr                (c0_ddr4_adr),
    .c0_ddr4_ba                 (c0_ddr4_ba),
    .c0_ddr4_bg                 (c0_ddr4_bg),
    .c0_ddr4_cke                (c0_ddr4_cke),
    .c0_ddr4_odt                (c0_ddr4_odt),
    .c0_ddr4_cs_n               (c0_ddr4_cs_n),
    .c0_ddr4_ck_t               (c0_ddr4_ck_t),
    .c0_ddr4_ck_c               (c0_ddr4_ck_c),
    .c0_ddr4_reset_n            (c0_ddr4_reset_n),   
    .c0_ddr4_parity             (c0_ddr4_parity),
    .c0_ddr4_dq                 (c0_ddr4_dq),
    .c0_ddr4_dqs_t              (c0_ddr4_dqs_t),
    .c0_ddr4_dqs_c              (c0_ddr4_dqs_c),
///////////ddr0 user interface
	.c0_ddr4_clk                (c0_ddr4_clk),
    .c0_ddr4_rst                (c0_ddr4_rst),
    .c0_init_complete           (c0_init_complete),
    .c0_ddr4_axi                (c0_ddr4_axi)

);




ddr1_driver inst_ddr1_driver(
	/*			DDR1 INTERFACE		*/
/////////ddr0 input clock
    .ddr1_sys_100M_p            (ddr1_sys_100M_p),
    .ddr1_sys_100M_n            (ddr1_sys_100M_n),
/////////ddr1 PHY interface
    .c1_ddr4_act_n              (c1_ddr4_act_n),
    .c1_ddr4_adr                (c1_ddr4_adr),
    .c1_ddr4_ba                 (c1_ddr4_ba),
    .c1_ddr4_bg                 (c1_ddr4_bg),    
    .c1_ddr4_cke                (c1_ddr4_cke),
    .c1_ddr4_odt                (c1_ddr4_odt),
    .c1_ddr4_cs_n               (c1_ddr4_cs_n),
    .c1_ddr4_ck_t               (c1_ddr4_ck_t),
    .c1_ddr4_ck_c               (c1_ddr4_ck_c),
    .c1_ddr4_reset_n            (c1_ddr4_reset_n),
    .c1_ddr4_parity             (c1_ddr4_parity),
    .c1_ddr4_dq                 (c1_ddr4_dq),
    .c1_ddr4_dqs_t              (c1_ddr4_dqs_t),
    .c1_ddr4_dqs_c              (c1_ddr4_dqs_c), 
///////////ddr1 user interface
	.c1_ddr4_clk                (c1_ddr4_clk),
    .c1_ddr4_rst                (c1_ddr4_rst),
    .c1_init_complete           (c1_init_complete),
	.c1_ddr4_axi                (c1_ddr4_axi)	

);

axi_clock_convert_warpper #(
    .WIDTH(512)
    ) inst_write0_clock_convert(
    .s_axi_aclk         (pcie_clk),
    .s_axi_aresetn      (user_rstn),

    .m_axi_aclk         (c0_ddr4_clk),
    .m_axi_aresetn      (~c0_ddr4_rst),

    .s_axi              (axis_mem_write0),
    .m_axi              (c0_ddr4_axi)
);

axi_register_slice_wrapper axi_register_slice_wrapper_inst(
    .aclk(c1_ddr4_clk),
    .aresetn(~c1_ddr4_rst),
    .s_axi(c1_ddr4_axi_reg),
    .m_axi(c1_ddr4_axi)
);


axi_clock_convert_warpper #(
    .WIDTH(512)
    ) inst_write1_clock_convert(
    .s_axi_aclk         (pcie_clk),
    .s_axi_aresetn      (user_rstn),

    .m_axi_aclk         (c1_ddr4_clk),
    .m_axi_aresetn      (~c1_ddr4_rst),

    .s_axi              (axis_mem_read0),
    .m_axi              (c1_ddr4_axi_reg)
);

mpi_reduce_control mpi_reduce_control_inst( 
    //user clock input
    .clk(pcie_clk),
    .rstn(user_rstn),
    
    //dma memory streams
    .m_axis_dma_write_cmd(axis_dma_write_cmd[0]),
    .m_axis_dma_write_data(axis_dma_write_data[0]),
    .m_axis_dma_read_cmd(axis_dma_read_cmd[0]),
    .s_axis_dma_read_data(axis_dma_read_data[0]),    

    //dma memory streams
    .m_axis_mem_read0(axis_mem_read0),
    // .m_axis_mem_read1(axis_mem_read1),
    .m_axis_mem_write0(axis_mem_write0),
    // .m_axis_mem_write1(axis_mem_write1),     

    //tcp app interface streams
    .app_axis_tcp_tx_meta(app_axis_tcp_tx_meta),
    .app_axis_tcp_tx_data(app_axis_tcp_tx_data),
    .app_axis_tcp_tx_status(app_axis_tcp_tx_status),    

    .app_axis_tcp_rx_meta(app_axis_tcp_rx_meta),
    .app_axis_tcp_rx_data(app_axis_tcp_rx_data),

    //control reg
    .control_reg(fpga_control_reg[271:256]),
    .status_reg(fpga_status_reg[287:256])

    
);

// mem_inf_transfer inst_mem_inf_transfer0(
//     .user_clk                   (pcie_clk),
//     .user_aresetn               (user_rstn),
//     .mem_clk                    (c0_ddr4_clk),
//     .mem_aresetn                (~c0_ddr4_rst),
    
//     /* USER INTERFACE */    
//     //memory access
//     //read cmd
//     .s_axis_mem_read_cmd        (axis_mem_read_cmd),
//     //read status
//     .m_axis_mem_read_status     (axis_mem_read_sts),
//     //read data stream
//     .m_axis_mem_read_data       (axis_mem_read_data),
    
//     //write cmd
//     .s_axis_mem_write_cmd       (axis_mem_write_cmd),
//     //write status
//     .m_axis_mem_write_status    (axis_mem_write_sts),
//     //write data stream
//     .s_axis_mem_write_data      (axis_mem_write_data),

//     .network_tx_mem             (c0_ddr4_axi)


//     );

//     mem_inf_transfer inst_mem_inf_transfer1(
//         .user_clk                   (pcie_clk),
//         .user_aresetn               (user_rstn),
//         .mem_clk                    (c1_ddr4_clk),
//         .mem_aresetn                (~c1_ddr4_rst),
        
//         /* USER INTERFACE */    
//         //memory access
//         //read cmd
//         .s_axis_mem_read_cmd        (axis_mem_read_cmd[1]),
//         //read status
//         .m_axis_mem_read_status     (axis_mem_read_sts[1]),
//         //read data stream
//         .m_axis_mem_read_data       (axis_mem_read_data[1]),
        
//         //write cmd
//         .s_axis_mem_write_cmd       (axis_mem_write_cmd[1]),
//         //write status
//         .m_axis_mem_write_status    (axis_mem_write_sts[1]),
//         //write data stream
//         .s_axis_mem_write_data      (axis_mem_write_data[1]),
    
//         .network_tx_mem             (c1_ddr4_axi)
    
    
//         );

    // // memory cmd streams
    //     axis_mem_cmd        axis_mem_dma_read_cmd();
    //     axis_mem_cmd        axis_mem_dma_write_cmd();
    //     // memory sts streams
    //     axis_mem_status     axis_mem_dma_read_sts();
    //     axis_mem_status     axis_mem_dma_write_sts();
    //     // memory data streams
    //     axi_stream          axis_mem_dma_read_data();
    //     axi_stream          axis_mem_dma_write_data();        


    //     axis_mem_cmd                axis_dma_ctrl_read_cmd();
    //     axi_stream                  axis_dma_ctrl_read_data();


    // dma_put_data_to_fpga dma_put_data_to_fpga_inst( 
    
    //     //user clock input
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),
    
    //     //DMA Commands
    //     .axis_dma_read_cmd          (axis_dma_ctrl_read_cmd),
    //     //DMA Data streams      
    //     .axis_dma_read_data         (axis_dma_ctrl_read_data),
        
    //     //tcp send
    //     .m_axis_mem_write_cmd        (axis_mem_dma_write_cmd),
    //     .s_axis_mem_write_sts        (axis_mem_dma_write_sts),
    //     .m_axis_mem_write_data       (axis_mem_dma_write_data),
    
    //     //control reg
    //     .s_axis_put_data_cmd         (axis_put_data_cmd),


    //     .control_reg                 (fpga_control_reg[207:192]),
    //     .status_reg                  ()
    
    // );

    // dma_get_data_from_fpga dma_get_data_from_fpga_inst( 

    //     //user clock input
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),
    
    //     //DMA Commands
    //     .axis_dma_write_cmd         (axis_dma_write_cmd[0]),
    
    //     //DMA Data streams      
    //     .axis_dma_write_data        (axis_dma_write_data[0]),
    
    //     //tcp send
    //     .m_axis_mem_read_cmd        (axis_mem_dma_read_cmd),
    //     .s_axis_mem_read_sts        (axis_mem_dma_read_sts),           
    //     .s_axis_mem_read_data       (axis_mem_dma_read_data),
    
    //     //control reg
    //     .s_axis_get_data_cmd        (axis_get_data_cmd),
    //     .control_reg                (fpga_control_reg[224:208]),
    //     .status_reg                 ()
    
    // );              
    




    // mpi_reduce_control inst_mpi_reduce_control( 

    //     //user clock input
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),
        
    //     //ddr memory streams
    //     .m_axis_mem_write_cmd       (axis_mem_write_cmd),
    //     .s_axis_mem_write_sts       (axis_mem_write_sts),
    //     .m_axis_mem_write_data      (axis_mem_write_data),
    //     .m_axis_mem_read_cmd        (axis_mem_read_cmd),
    //     .s_axis_mem_read_sts        (axis_mem_read_sts),
    //     .s_axis_mem_read_data       (axis_mem_read_data),    
    
    //     //dma memory streams
    //     .s_axis_mem_dma_read_cmd    (axis_mem_dma_read_cmd),
    //     .s_axis_mem_dma_write_cmd   (axis_mem_dma_write_cmd),
    //     .m_axis_mem_dma_read_sts    (axis_mem_dma_read_sts),
    //     .m_axis_mem_dma_write_sts   (axis_mem_dma_write_sts),
    //     .m_axis_mem_dma_read_data   (axis_mem_dma_read_data),
    //     .s_axis_mem_dma_write_data  (axis_mem_dma_write_data), 
    //     //DMA Commands
    //     .m_axis_dma_read_cmd        (axis_dma_read_cmd[0]),    
    //     .axis_dma_read_data         (axis_dma_read_data[0]),  
        
    //     //
    //     .axis_dma_ctrl_read_cmd     (axis_dma_ctrl_read_cmd),
    //     .axis_dma_ctrl_read_data    (axis_dma_ctrl_read_data),        
    
    //     //tcp app interface streams
    //     .app_axis_tcp_tx_meta       (app_axis_tcp_tx_meta),
    //     .app_axis_tcp_tx_data       (app_axis_tcp_tx_data),
    //     .app_axis_tcp_tx_status     (app_axis_tcp_tx_status),    
    
    //     .app_axis_tcp_rx_meta       (app_axis_tcp_rx_meta),
    //     .app_axis_tcp_rx_data       (app_axis_tcp_rx_data),
    
    //     //control reg
    //     .dma_base_addr              ({fpga_control_reg[193],fpga_control_reg[192]}),
    //     .control_reg                (fpga_control_reg[271:256]),
    //     .status_reg                 (fpga_status_reg[287:256])
    
        
    //     );

// inference inference_inst (
//     .clk(pcie_clk),
//     .rstn(user_rstn),
   
    
//     //dma memory streams
//     .axis_dma_read_cmd(axis_dma_read_cmd[0]),
//     .axis_dma_read_data(axis_dma_read_data[0]),    
    

//     //tcp app interface streams
//     .m_axis_tx_metadata(app_axis_tcp_tx_meta),
//     .m_axis_tx_data(app_axis_tcp_tx_data),
//     .s_axis_tx_status(app_axis_tcp_tx_status),    

//     .s_axis_rx_metadata(app_axis_tcp_rx_meta),
//     .s_axis_rx_data(app_axis_tcp_rx_data),


//    //control reg
//     .control_reg                    (fpga_control_reg[303:288]),
//     .status_reg                     (fpga_status_reg[305:290])     

// );


////////////////////dma benchmark//////////
// dma_write_engine dma_write_engine_inst( 
//     .clk                        (pcie_clk),
//     .rstn                       (user_rstn),
    
//     //DMA Commands
//     .m_axis_dma_write_cmd       (axis_dma_write_cmd[0]),

//     //DMA Data streams      
//     .m_axis_dma_write_data      (axis_dma_write_data[0]),
    
//     .control_reg                (fpga_control_reg[47:32]),
//     .status_reg                 (fpga_status_reg[79:64])

//     );

    // dma_read_engine dma_read_engine_inst( 
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),
        
    //     //DMA Commands
    //     .m_axis_dma_read_cmd       (axis_dma_read_cmd[0]),
    
    //     //DMA Data streams      
    //     .s_axis_dma_read_data      (axis_dma_read_data[0]),
        
    //     .control_reg                (fpga_control_reg[47:32]),
    //     .status_reg                 (fpga_status_reg[95:80])
    
    // );

///////////////////tcp benchmark//////////////////////


    // tcp_send_engine tcp_send_engine_inst( 
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),
        
    //     //tcp interface
    //     .m_axis_tx_metadata         (app_axis_tcp_tx_meta),
    //     .m_axis_tx_data             (app_axis_tcp_tx_data),
    //     .s_axis_tx_status           (app_axis_tcp_tx_status),

    //     .s_axis_rx_metadata         (app_axis_tcp_rx_meta),
        
    //     .control_reg                (fpga_control_reg[63:48]),
    //     .status_reg                 (fpga_status_reg[103:96])
    
    // );


    // tcp_recv_engine tcp_recv_engine_inst( 
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),
        
    //     //tcp interface
    //     // .s_axis_notifications       (axis_tcp_notification),
    //     // .m_axis_read_package        (axis_tcp_read_pkg),
    //     // .s_axis_rx_metadata         (app_axis_tcp_rx_meta),
    //     .s_axis_rx_metadata_ready   (app_axis_tcp_rx_meta.ready),
    //     .s_axis_rx_metadata_valid   (app_axis_tcp_rx_meta.valid),
    //     .s_axis_rx_data             (app_axis_tcp_rx_data),
        
    //     .control_reg                (fpga_control_reg[63:48]),
    //     .status_reg                 (fpga_status_reg[111:104])
    
    // );


    // tcp_latency_engine inst_tcp_latency_engine( 
    //     .clk                        (pcie_clk),
    //     .rstn                       (user_rstn),

    //     .m_axis_tx_metadata         (axis_tcp_tx_meta),
    //     .m_axis_tx_data             (axis_tcp_tx_data),
    //     .s_axis_tx_status           (axis_tcp_tx_status),
	
    //     .s_axis_notifications       (axis_tcp_notification),
    //     .m_axis_read_package        (axis_tcp_read_pkg),
    //     .s_axis_rx_metadata         (axis_tcp_rx_meta),
    //     .s_axis_rx_data             (axis_tcp_rx_data),

    
    //     .control_reg                (fpga_control_reg[63:48]),
    //     .status_reg                 (fpga_status_reg[103:96]) 

    // );
////////////////////////////////hbm_benchmark

// mem_benchmark inst_mem_benchmark(
//     .clk                                (pcie_clk),   //should be 450MHz, 
//     .rstn                              (user_rstn), //negative reset,   

//     .hbm_clk                            (hbm_clk),
//     .hbm_rstn                           (hbm_rstn),

//     .hbm_axi                            (hbm_axi),    

// /////////////////////////
//     .fpga_control                       (fpga_control_reg[79:64]),
//     .fpga_status                        (fpga_status_reg[207:192])   

// );

//////////////////////////barrier

        // barrier inst_barrier( 

        //     //user clock input
        //     .clk(pcie_clk),
        //     .rstn(user_rstn),
            
        //     //tcp send
        //     .m_axis_tx_metadata(app_axis_tcp_tx_meta),
        //     .m_axis_tx_data(app_axis_tcp_tx_data),
        //     .s_axis_tx_status(app_axis_tcp_tx_status),
        //     //tcp recv   
        //     .s_axis_rx_metadata(app_axis_tcp_rx_meta),
        //     .s_axis_rx_data(app_axis_tcp_rx_data),
        
        //     //control reg
        //     .control_reg(fpga_control_reg[335:320]),
        //     .status_reg(fpga_status_reg[327:320])
        
            
        //     );        

endmodule
