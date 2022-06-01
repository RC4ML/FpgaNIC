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


module ddr0_driver( 
	/*			DDR0 INTERFACE		*/
/////////ddr0 input clock
	input						ddr0_sys_100M_p,
	input						ddr0_sys_100M_n,
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
///////////ddr0 user interface
	output						c0_ddr4_clk,
  	output						c0_ddr4_rst,
  	output            			c0_init_complete,
	axi_mm.slave                c0_ddr4_axi

    );

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////            ddr
//////////////////////////////////////////////////////////////////////////////////////////// 

	wire 						ddr0_sys_100M,DDR0_sys_clk;

    IBUFDS #(
      .IBUF_LOW_PWR("TRUE")     // Low power="TRUE", Highest performance="FALSE" 
   ) IBUFDS0_inst (
      .O(ddr0_sys_100M),  // Buffer output
      .I(ddr0_sys_100M_p),  // Diff_p buffer input (connect directly to top-level port)
      .IB(ddr0_sys_100M_n) // Diff_n buffer input (connect directly to top-level port)
   );
 
   
     BUFG BUFG0_inst (
      .O(DDR0_sys_clk), // 1-bit output: Clock output
      .I(ddr0_sys_100M)  // 1-bit input: Clock input
   ); 
   
   
ddr4_0 u_ddr4_0
  (
   .sys_rst           (1'b0),

   .c0_sys_clk_i           (DDR0_sys_clk),
   .c0_init_calib_complete (c0_init_complete),
   .c0_ddr4_act_n          (c0_ddr4_act_n),
   .c0_ddr4_adr            (c0_ddr4_adr),
   .c0_ddr4_ba             (c0_ddr4_ba),
   .c0_ddr4_bg             (c0_ddr4_bg),
   .c0_ddr4_cke            (c0_ddr4_cke),
   .c0_ddr4_odt            (c0_ddr4_odt),
   .c0_ddr4_cs_n           (c0_ddr4_cs_n),
   .c0_ddr4_ck_t           (c0_ddr4_ck_t),
   .c0_ddr4_ck_c           (c0_ddr4_ck_c),
   .c0_ddr4_reset_n        (c0_ddr4_reset_n),

   .c0_ddr4_parity         (c0_ddr4_parity),
   .c0_ddr4_dq             (c0_ddr4_dq),
   .c0_ddr4_dqs_c          (c0_ddr4_dqs_c),
   .c0_ddr4_dqs_t          (c0_ddr4_dqs_t),

   .c0_ddr4_ui_clk                (c0_ddr4_clk),
   .c0_ddr4_ui_clk_sync_rst       (c0_ddr4_rst),
   .addn_ui_clkout1                            (),
   .dbg_clk                                    (),
     // AXI CTRL port
     .c0_ddr4_s_axi_ctrl_awvalid       (1'b0),
     .c0_ddr4_s_axi_ctrl_awready       (),
     .c0_ddr4_s_axi_ctrl_awaddr        (32'b0),
     // Slave Interface Write Data Ports
     .c0_ddr4_s_axi_ctrl_wvalid        (1'b0),
     .c0_ddr4_s_axi_ctrl_wready        (),
     .c0_ddr4_s_axi_ctrl_wdata         (32'b0),
     // Slave Interface Write Response Ports
     .c0_ddr4_s_axi_ctrl_bvalid        (),
     .c0_ddr4_s_axi_ctrl_bready        (1'b1),
     .c0_ddr4_s_axi_ctrl_bresp         (),
     // Slave Interface Read Address Ports
     .c0_ddr4_s_axi_ctrl_arvalid       (1'b0),
     .c0_ddr4_s_axi_ctrl_arready       (),
     .c0_ddr4_s_axi_ctrl_araddr        (32'b0),
     // Slave Interface Read Data Ports
     .c0_ddr4_s_axi_ctrl_rvalid        (),
     .c0_ddr4_s_axi_ctrl_rready        (1'b1),
     .c0_ddr4_s_axi_ctrl_rdata         (),
     .c0_ddr4_s_axi_ctrl_rresp         (),
     // Interrupt output
     .c0_ddr4_interrupt                (),
  // Slave Interface Write Address Ports
  .c0_ddr4_aresetn                     (~c0_ddr4_rst),
  .c0_ddr4_s_axi_awid                  (c0_ddr4_axi.awid),
  .c0_ddr4_s_axi_awaddr                (c0_ddr4_axi.awaddr),
  .c0_ddr4_s_axi_awlen                 (c0_ddr4_axi.awlen),
  .c0_ddr4_s_axi_awsize                (c0_ddr4_axi.awsize),
  .c0_ddr4_s_axi_awburst               (c0_ddr4_axi.awburst),
  .c0_ddr4_s_axi_awlock                (c0_ddr4_axi.awlock),
  .c0_ddr4_s_axi_awcache               (c0_ddr4_axi.awcache),
  .c0_ddr4_s_axi_awprot                (c0_ddr4_axi.awprot),
  .c0_ddr4_s_axi_awqos                 (c0_ddr4_axi.awqos),
  .c0_ddr4_s_axi_awvalid               (c0_ddr4_axi.awvalid),
  .c0_ddr4_s_axi_awready               (c0_ddr4_axi.awready),
  // Slave Interface Write Data Ports
  .c0_ddr4_s_axi_wdata                 (c0_ddr4_axi.wdata),
  .c0_ddr4_s_axi_wstrb                 (c0_ddr4_axi.wstrb),
  .c0_ddr4_s_axi_wlast                 (c0_ddr4_axi.wlast),
  .c0_ddr4_s_axi_wvalid                (c0_ddr4_axi.wvalid),
  .c0_ddr4_s_axi_wready                (c0_ddr4_axi.wready),
  // Slave Interface Write Response Ports
  .c0_ddr4_s_axi_bid                   (c0_ddr4_axi.bid),
  .c0_ddr4_s_axi_bresp                 (c0_ddr4_axi.bresp),
  .c0_ddr4_s_axi_bvalid                (c0_ddr4_axi.bvalid),
  .c0_ddr4_s_axi_bready                (c0_ddr4_axi.bready),
  // Slave Interface Read Address Ports
  .c0_ddr4_s_axi_arid                  (c0_ddr4_axi.arid),
  .c0_ddr4_s_axi_araddr                (c0_ddr4_axi.araddr),
  .c0_ddr4_s_axi_arlen                 (c0_ddr4_axi.arlen),
  .c0_ddr4_s_axi_arsize                (c0_ddr4_axi.arsize),
  .c0_ddr4_s_axi_arburst               (c0_ddr4_axi.arburst),
  .c0_ddr4_s_axi_arlock                (c0_ddr4_axi.arlock),
  .c0_ddr4_s_axi_arcache               (c0_ddr4_axi.arcache),
  .c0_ddr4_s_axi_arprot                (c0_ddr4_axi.arprot),
  .c0_ddr4_s_axi_arqos                 (c0_ddr4_axi.arqos),
  .c0_ddr4_s_axi_arvalid               (c0_ddr4_axi.arvalid),
  .c0_ddr4_s_axi_arready               (c0_ddr4_axi.arready),
  // Slave Interface Read Data Ports
  .c0_ddr4_s_axi_rid                   (c0_ddr4_axi.rid),
  .c0_ddr4_s_axi_rdata                 (c0_ddr4_axi.rdata),
  .c0_ddr4_s_axi_rresp                 (c0_ddr4_axi.rresp),
  .c0_ddr4_s_axi_rlast                 (c0_ddr4_axi.rlast),
  .c0_ddr4_s_axi_rvalid                (c0_ddr4_axi.rvalid),
  .c0_ddr4_s_axi_rready                (c0_ddr4_axi.rready),
  
  // Debug Port
  .dbg_bus         ()                                             

  ); 



endmodule


module ddr1_driver( 
	/*			DDR1 INTERFACE		*/
/////////ddr0 input clock
	input						ddr1_sys_100M_p,
	input						ddr1_sys_100M_n,
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
///////////ddr1 user interface
	output						c1_ddr4_clk,
  	output						c1_ddr4_rst,
  	output            			c1_init_complete,
	axi_mm.slave                c1_ddr4_axi	

    );

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////            ddr
//////////////////////////////////////////////////////////////////////////////////////////// 

	wire 						ddr1_sys_100M,DDR1_sys_clk;
 
    IBUFDS #(
      .IBUF_LOW_PWR("TRUE")     // Low power="TRUE", Highest performance="FALSE" 
   ) IBUFDS1_inst (
      .O(ddr1_sys_100M),  // Buffer output
      .I(ddr1_sys_100M_p),  // Diff_p buffer input (connect directly to top-level port)
      .IB(ddr1_sys_100M_n) // Diff_n buffer input (connect directly to top-level port)
   );
 
   
     BUFG BUFG1_inst (
      .O(DDR1_sys_clk), // 1-bit output: Clock output
      .I(ddr1_sys_100M)  // 1-bit input: Clock input
   );
   


ddr4_1 u_ddr4_1
  (
   .sys_rst           (1'b0),

   .c0_sys_clk_i           (DDR1_sys_clk),
   .c0_init_calib_complete (c1_init_complete),
   .c0_ddr4_act_n          (c1_ddr4_act_n),
   .c0_ddr4_adr            (c1_ddr4_adr),
   .c0_ddr4_ba             (c1_ddr4_ba),
   .c0_ddr4_bg             (c1_ddr4_bg),
   .c0_ddr4_cke            (c1_ddr4_cke),
   .c0_ddr4_odt            (c1_ddr4_odt),
   .c0_ddr4_cs_n           (c1_ddr4_cs_n),
   .c0_ddr4_ck_t           (c1_ddr4_ck_t),
   .c0_ddr4_ck_c           (c1_ddr4_ck_c),
   .c0_ddr4_reset_n        (c1_ddr4_reset_n),

   .c0_ddr4_parity         (c1_ddr4_parity),
   .c0_ddr4_dq             (c1_ddr4_dq),
   .c0_ddr4_dqs_c          (c1_ddr4_dqs_c),
   .c0_ddr4_dqs_t          (c1_ddr4_dqs_t),

   .c0_ddr4_ui_clk                (c1_ddr4_clk),
   .c0_ddr4_ui_clk_sync_rst       (c1_ddr4_rst),
   .addn_ui_clkout1                            (),
   .dbg_clk                                    (),
     // AXI CTRL port
     .c0_ddr4_s_axi_ctrl_awvalid       (1'b0),
     .c0_ddr4_s_axi_ctrl_awready       (),
     .c0_ddr4_s_axi_ctrl_awaddr        (32'b0),
     // Slave Interface Write Data Ports
     .c0_ddr4_s_axi_ctrl_wvalid        (1'b0),
     .c0_ddr4_s_axi_ctrl_wready        (),
     .c0_ddr4_s_axi_ctrl_wdata         (32'b0),
     // Slave Interface Write Response Ports
     .c0_ddr4_s_axi_ctrl_bvalid        (),
     .c0_ddr4_s_axi_ctrl_bready        (1'b1),
     .c0_ddr4_s_axi_ctrl_bresp         (),
     // Slave Interface Read Address Ports
     .c0_ddr4_s_axi_ctrl_arvalid       (1'b0),
     .c0_ddr4_s_axi_ctrl_arready       (),
     .c0_ddr4_s_axi_ctrl_araddr        (32'b0),
     // Slave Interface Read Data Ports
     .c0_ddr4_s_axi_ctrl_rvalid        (),
     .c0_ddr4_s_axi_ctrl_rready        (1'b1),
     .c0_ddr4_s_axi_ctrl_rdata         (),
     .c0_ddr4_s_axi_ctrl_rresp         (),
     // Interrupt output
     .c0_ddr4_interrupt                (),
  // Slave Interface Write Address Ports
  .c0_ddr4_aresetn                     (~c1_ddr4_rst),
  .c0_ddr4_s_axi_awid                  (c1_ddr4_axi.awid),
  .c0_ddr4_s_axi_awaddr                (c1_ddr4_axi.awaddr),
  .c0_ddr4_s_axi_awlen                 (c1_ddr4_axi.awlen),
  .c0_ddr4_s_axi_awsize                (c1_ddr4_axi.awsize),
  .c0_ddr4_s_axi_awburst               (c1_ddr4_axi.awburst),
  .c0_ddr4_s_axi_awlock                (c1_ddr4_axi.awlock),
  .c0_ddr4_s_axi_awcache               (c1_ddr4_axi.awcache),
  .c0_ddr4_s_axi_awprot                (c1_ddr4_axi.awprot),
  .c0_ddr4_s_axi_awqos                 (c1_ddr4_axi.awqos),
  .c0_ddr4_s_axi_awvalid               (c1_ddr4_axi.awvalid),
  .c0_ddr4_s_axi_awready               (c1_ddr4_axi.awready),
  // Slave Interface Write Data Ports           
  .c0_ddr4_s_axi_wdata                 (c1_ddr4_axi.wdata),
  .c0_ddr4_s_axi_wstrb                 (c1_ddr4_axi.wstrb),
  .c0_ddr4_s_axi_wlast                 (c1_ddr4_axi.wlast),
  .c0_ddr4_s_axi_wvalid                (c1_ddr4_axi.wvalid),
  .c0_ddr4_s_axi_wready                (c1_ddr4_axi.wready),
  // Slave Interface Write Response Ports       
  .c0_ddr4_s_axi_bid                   (c1_ddr4_axi.bid),
  .c0_ddr4_s_axi_bresp                 (c1_ddr4_axi.bresp),
  .c0_ddr4_s_axi_bvalid                (c1_ddr4_axi.bvalid),
  .c0_ddr4_s_axi_bready                (c1_ddr4_axi.bready),
  // Slave Interface Read Address Ports
  .c0_ddr4_s_axi_arid                  (c1_ddr4_axi.arid),
  .c0_ddr4_s_axi_araddr                (c1_ddr4_axi.araddr),
  .c0_ddr4_s_axi_arlen                 (c1_ddr4_axi.arlen),
  .c0_ddr4_s_axi_arsize                (c1_ddr4_axi.arsize),
  .c0_ddr4_s_axi_arburst               (c1_ddr4_axi.arburst),
  .c0_ddr4_s_axi_arlock                (c1_ddr4_axi.arlock),
  .c0_ddr4_s_axi_arcache               (c1_ddr4_axi.arcache),
  .c0_ddr4_s_axi_arprot                (c1_ddr4_axi.arprot),
  .c0_ddr4_s_axi_arqos                 (c1_ddr4_axi.arqos),
  .c0_ddr4_s_axi_arvalid               (c1_ddr4_axi.arvalid),
  .c0_ddr4_s_axi_arready               (c1_ddr4_axi.arready),
  // Slave Interface Read Data Ports
  .c0_ddr4_s_axi_rid                   (c1_ddr4_axi.rid),
  .c0_ddr4_s_axi_rdata                 (c1_ddr4_axi.rdata),
  .c0_ddr4_s_axi_rresp                 (c1_ddr4_axi.rresp),
  .c0_ddr4_s_axi_rlast                 (c1_ddr4_axi.rlast),
  .c0_ddr4_s_axi_rvalid                (c1_ddr4_axi.rvalid),
  .c0_ddr4_s_axi_rready                (c1_ddr4_axi.rready),
  
  // Debug Port
  .dbg_bus         ()                                             

  ); 



endmodule