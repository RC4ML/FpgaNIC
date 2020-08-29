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
`include"sgd_defines.vh"

module xdma_top_example( 
    output wire[15 : 0] pcie_tx_p,
    output wire[15 : 0] pcie_tx_n,
    input wire[15 : 0]  pcie_rx_p,
    input wire[15 : 0]  pcie_rx_n,

    input wire				sys_clk_p,
    input wire				sys_clk_n,
    input wire				sys_rst_n   
    );
    




// DMA Signals
axis_mem_cmd    axis_dma_read_cmd();
axis_mem_cmd    axis_dma_write_cmd();
axi_stream      axis_dma_read_data();
axi_stream      axis_dma_write_data();



wire[511:0][31:0]     fpga_control_reg;
wire[511:0][31:0]     fpga_status_reg; 

wire[31:0][511:0]     bypass_control_reg;
wire[31:0][511:0]     bypass_status_reg;

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
	.fpga_status_reg                (fpga_status_reg)

`ifdef XDMA_BYPASS		
    // bypass register
	,.bypass_control_reg 			(bypass_control_reg),
	.bypass_status_reg  			(bypass_status_reg)
`endif

);





/////////////////////////dma debug///////////

//test dma speed
reg [31:0]		error_cnt;
reg [31:0]		error_index;


  wire              user_clk;
  reg            	read_start;
  reg				read_start_d;
  reg            	write_start;
  reg				write_start_d;
  reg [31:0]		axis_dma_write_data_cnt;
  reg [31:0]		axis_dma_write_data_length;
  reg [31:0]		axis_dma_read_data_cnt;
  reg [31:0]		axis_dma_read_data_length;
  reg [31:0]		ops;

  reg [31:0]		read_cnt;
  reg [31:0]		write_cnt;
  reg 				read_cnt_en;
  reg 				write_cnt_en;
  reg [31:0]		wr_op_cnt;
  reg [31:0]		rd_op_cnt;
  reg [31:0]		wr_op_data_cnt;
  reg [31:0]		rd_op_data_cnt;  

assign user_clk = pcie_clk;
assign user_aresetn = pcie_aresetn;

always @(posedge user_clk)begin
	read_start_d <= read_start;
	write_start_d <= write_start;
end




always @(posedge user_clk)begin
	if(~pcie_aresetn)
		axis_dma_read_cmd.valid <= 1'b0;
	else if(read_start && ~read_start_d)
		axis_dma_read_cmd.valid <= 1'b1;
	else if(axis_dma_read_cmd.valid && axis_dma_read_cmd.ready)
		axis_dma_read_cmd.valid <= 1'b0;
	else 
		axis_dma_read_cmd.valid <= axis_dma_read_cmd.valid;
end

always @(posedge user_clk)begin
	if(~pcie_aresetn)
		axis_dma_write_cmd.valid <= 1'b0;
	else if(write_start && ~write_start_d)
		axis_dma_write_cmd.valid <= 1'b1;
	else if(axis_dma_write_cmd.valid && axis_dma_write_cmd.ready)
		axis_dma_write_cmd.valid <= 1'b0;
	else 
		axis_dma_write_cmd.valid <= axis_dma_write_cmd.valid;
end

always @(posedge user_clk)begin
	if(~pcie_aresetn)
		axis_dma_write_data_cnt <= 1'b0;
	else if(axis_dma_write_data.last)
		axis_dma_write_data_cnt <= 1'b0;
	else if(axis_dma_write_data.valid && axis_dma_write_data.ready)
		axis_dma_write_data_cnt <= axis_dma_write_data_cnt + 1;    
	else
		axis_dma_write_data_cnt <= axis_dma_write_data_cnt;
end

always @(posedge user_clk)begin
	axis_dma_write_data_length <= (axis_dma_write_cmd.length>>6) - 1;
end


always @(posedge user_clk)begin
	if(~pcie_aresetn)
		axis_dma_read_data_cnt <= 1'b0;
	else if((axis_dma_read_data_cnt == axis_dma_read_data_length) && axis_dma_read_data.valid && axis_dma_read_data.ready)
		axis_dma_read_data_cnt <= 1'b0;
	else if(axis_dma_read_data.valid && axis_dma_read_data.ready)
		axis_dma_read_data_cnt <= axis_dma_read_data_cnt + 1;    
	else
		axis_dma_read_data_cnt <= axis_dma_read_data_cnt;
end

always @(posedge user_clk)begin
	axis_dma_read_data_length <= (axis_dma_read_cmd.length>>6) - 1;
end

assign axis_dma_read_data.ready = 1'b1;
assign axis_dma_write_data.valid = 1'b1;
assign axis_dma_write_data.keep = 64'hffff_ffff_ffff_ffff;
assign axis_dma_write_data.data = axis_dma_write_data_cnt;
assign axis_dma_write_data.last = axis_dma_write_data.valid && axis_dma_write_data.ready && (axis_dma_write_data_cnt == axis_dma_write_data_length);


//vio_0 your_instance_name (
//  .clk(user_clk),                // input wire clk
//  .probe_out0(axis_dma_read_cmd.address),  // output wire [63 : 0] probe_out0
//  .probe_out1(axis_dma_read_cmd.length),  // output wire [31 : 0] probe_out1
//  .probe_out2(read_start),  // output wire [0 : 0] probe_out2
//  .probe_out3(axis_dma_write_cmd.address),  // output wire [63 : 0] probe_out3
//  .probe_out4(axis_dma_write_cmd.length),  // output wire [31 : 0] probe_out4
//  .probe_out5(write_start)  // output wire [0 : 0] probe_out5
//);


always @(posedge user_clk)begin
	if(~pcie_aresetn)
		error_cnt <= 1'b0;
	else if((axis_dma_read_data_cnt != axis_dma_read_data.data[479:448]) && axis_dma_read_data.valid && axis_dma_read_data.ready)
		error_cnt <= error_cnt + 1'b1;   
	else
		error_cnt <= error_cnt;
end

always @(posedge user_clk)begin
	if(~pcie_aresetn)
		error_index <= 1'b0;
	else if((axis_dma_read_data_cnt != axis_dma_read_data.data[479:448]) && axis_dma_read_data.valid && axis_dma_read_data.ready)
		error_index <= axis_dma_read_data_cnt;   
	else
		error_index <= error_index;
end

//ila_0 inst_ila_0 (
//	.clk(user_clk), // input wire clk


//	.probe0(axis_dma_read_cmd.valid), // input wire [0:0]  probe0  
//	.probe1(axis_dma_read_cmd.ready), // input wire [0:0]  probe1 
//	.probe2(axis_dma_write_cmd.valid), // input wire [0:0]  probe2 
//	.probe3(axis_dma_write_cmd.ready), // input wire [0:0]  probe3 
//	.probe4(axis_dma_read_data.data), // input wire [511:0]  probe4 
//	.probe5(read_cnt), // input wire [31:0]  probe5 
//	.probe6(axis_dma_read_data_cnt), // input wire [31:0]  probe6 
//	.probe7(axis_dma_read_data.valid), // input wire [0:0]  probe7 
//	.probe8(axis_dma_write_data.ready), // input wire [0:0]  probe8 
//	.probe9(axis_dma_write_data.data), // input wire [511:0]  probe9 
//	.probe10(write_cnt), // input wire [31:0]  probe10 
//	.probe11(axis_dma_write_data_cnt), // input wire [31:0]  probe11
//	.probe12(error_cnt), // input wire [31:0]  probe12 
//	.probe13(error_index) // input wire [31:0]  probe13
//);


endmodule
