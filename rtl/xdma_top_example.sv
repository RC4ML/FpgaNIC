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
axis_mem_cmd    axis_dma_read_cmd[4]();
axis_mem_cmd    axis_dma_write_cmd[4]();
axi_stream      axis_dma_read_data[4]();
axi_stream      axis_dma_write_data[4]();




wire[511:0][31:0]     fpga_control_reg;
wire[511:0][31:0]     fpga_status_reg; 

wire[31:0][511:0]     bypass_control_reg;
wire[31:0][511:0]     bypass_status_reg;


//reset

reg 					reset,reset_r;
reg[7:0]				reset_cnt;
reg 					user_rstn;		

always @(posedge pcie_clk)begin
	reset				<= fpga_control_reg[0][0];
	reset_r				<= reset;
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

assign user_rstn = pcie_aresetn & reset_cnt[7];



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
    .user_aresetn					(user_rstn),

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

genvar i;
generate
	for(i = 0; i < 4; i = i + 1) begin

dma_data_transfer#(
    .PAGE_SIZE (2*1024*1024)	,
    .PAGE_NUM  (109)		,
    .CTRL_NUM  (1024)		 
)dma_data_transfer_inst( 

    //user clock input
    .clk							(pcie_clk),
    .rstn							(user_rstn),

    //DMA Commands
    .axis_dma_read_cmd				(axis_dma_read_cmd[i]),
    .axis_dma_write_cmd				(axis_dma_write_cmd[i]),

    //DMA Data streams      
    .axis_dma_write_data			(axis_dma_write_data[i]),
    .axis_dma_read_data				(axis_dma_read_data[i]),

    //control reg
    .transfer_base_addr				({fpga_control_reg[41+i*8],fpga_control_reg[40+i*8]}),

    .transfer_start_page			(fpga_control_reg[42+i*8]),                      
    .transfer_length				(fpga_control_reg[43+i*8]),
    .transfer_offset				(fpga_control_reg[44+i*8]),
    .work_page_size					(fpga_control_reg[45+i*8]),
    .transfer_start					(fpga_control_reg[46][i]),
	.gpu_read_count					(fpga_control_reg[47+i*8]),
	.gpu_write_count				(fpga_status_reg[60+i])

    );

	end
endgenerate

















///////////////////////dma_ex debug//////////

// 	localparam [3:0]		IDLE 			= 4'b0001,
// 							WRITE_CMD		= 4'b0010,
// 							WRITE_DATA		= 4'b0100;
	
// 	reg [3:0]								wr_state;							

// 	reg 									wr_start_r,wr_start_rr;
	reg 									rd_start_r,rd_start_rr;
	reg [31:0]								data_cnt,rd_data_cnt;
	reg [31:0]								offset;
	reg [31:0]								data_cnt_minus;

// 	reg 									wr_th_en;
// 	reg [31:0]								wr_th_sum;
	reg 									rd_th_en;
	reg [31:0]								rd_th_sum;
	reg 									rd_lat_en;
	reg [31:0]								rd_lat_sum;			

	reg [31:0]		                    	error_cnt;
	reg [31:0]		                    	error_index;
	reg 									error_flag,error_flag_r;		

// 	assign	axis_dma_write_cmd[0].address	= {fpga_control_reg[33],fpga_control_reg[32]};
// 	assign	axis_dma_write_cmd[0].length	= fpga_control_reg[34]; 
// 	assign 	axis_dma_write_cmd[0].valid		= (wr_state == WRITE_CMD); 	

	always @(posedge pcie_clk)begin
		// wr_start_r							<= fpga_control_reg[36][0];
		// wr_start_rr							<= wr_start_r;
		rd_start_r							<= fpga_control_reg[36][1];
		rd_start_rr							<= rd_start_r;		
	end

	always @(posedge pcie_clk)begin
		data_cnt_minus						<= (fpga_control_reg[34]>>>6)-1;
		offset								<= fpga_control_reg[35];
	end

// 	always @(posedge pcie_clk)begin
// 		if(~pcie_aresetn)begin
// 			data_cnt 						<= 1'b0;
// 		end
// 		else if(axis_dma_write_data[0].last)begin
// 			data_cnt						<= 1'b0;
// 		end
// 		else if(axis_dma_write_data[0].ready & axis_dma_write_data[0].valid)begin
// 			data_cnt						<= data_cnt + 1'b1;
// 		end
// 		else begin
// 			data_cnt						<= data_cnt;
// 		end		
// 	end

// 	assign axis_dma_write_data[0].valid		= (wr_state == WRITE_DATA);
// 	assign axis_dma_write_data[0].keep		= 64'hffff_ffff_ffff_ffff;
// 	assign axis_dma_write_data[0].last		= (data_cnt == data_cnt_minus) && axis_dma_write_data[0].ready && axis_dma_write_data[0].valid;
// 	assign axis_dma_write_data[0].data		= data_cnt + offset;

// 	always @(posedge pcie_clk)begin
// 		if(~pcie_aresetn)begin
// 			wr_state						<= IDLE;
// 		end
// 		else begin
// 			case(wr_state)
// 				IDLE:begin
// 					if(wr_start_r & ~wr_start_rr)begin
// 						wr_state			<= WRITE_CMD;
// 					end
// 					else begin
// 						wr_state			<= IDLE;
// 					end
// 				end
// 				WRITE_CMD:begin
// 					if(axis_dma_write_cmd[0].ready & axis_dma_write_cmd[0].valid)begin
// 						wr_state			<= WRITE_DATA;
// 					end
// 					else begin
// 						wr_state			<= WRITE_CMD;
// 					end
// 				end
// 				WRITE_DATA:begin
// 					if(axis_dma_write_data[0].last)begin
// 						wr_state			<= IDLE;
// 					end	
// 					else begin
// 						wr_state			<= WRITE_DATA;
// 					end
// 				end
// 			endcase
// 		end
// 	end

// 	always@(posedge pcie_clk)begin
// 		if(~pcie_aresetn)begin
// 			wr_th_en						<= 1'b0;
// 		end  
// 		else if(axis_dma_write_data[0].last)begin
// 			wr_th_en						<= 1'b0;
// 		end
// 		else if(axis_dma_write_cmd[0].ready & axis_dma_write_cmd[0].valid)begin
// 			wr_th_en						<= 1'b1;
// 		end		
// 		else begin
// 			wr_th_en						<= wr_th_en;
// 		end
// 	end

	
// 	always@(posedge pcie_clk)begin
// 		if(~pcie_aresetn)begin
// 			wr_th_sum						<= 32'b0;
// 		end
// 		else if(wr_start_r & ~wr_start_rr)begin
// 			wr_th_sum						<= 32'b0;
// 		end 
// 		else if(wr_th_en)begin
// 			wr_th_sum						<= wr_th_sum + 1'b1;
// 		end
// 		else begin
// 			wr_th_sum						<= wr_th_sum;
// 		end
// 	end

// 	assign fpga_status_reg[54] = wr_th_sum;
// //////////////////////////////////////////////////////read

	reg 									dma_read_cmd_valid;

	assign	axis_dma_read_cmd[0].address	= {fpga_control_reg[33],fpga_control_reg[32]};
	assign	axis_dma_read_cmd[0].length		= fpga_control_reg[34]; 
	assign 	axis_dma_read_cmd[0].valid		= dma_read_cmd_valid; 

	assign  axis_dma_read_data[1].ready		= 1;
	assign  axis_dma_read_data[0].ready		= 1;
	assign  axis_dma_read_data[2].ready		= 1;
	assign  axis_dma_read_data[3].ready		= 1;

	always @(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			dma_read_cmd_valid				<= 1'b0;
		end
		else if(rd_start_r & ~rd_start_rr)begin
			dma_read_cmd_valid				<= 1'b1;
		end
		else if(axis_dma_read_cmd[0].valid & axis_dma_read_cmd[0].ready)begin
			dma_read_cmd_valid				<= 1'b0;
		end
		else begin
			dma_read_cmd_valid				<= dma_read_cmd_valid;
		end		
	end

	always @(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_data_cnt 					<= 1'b0;
		end
		else if(dma_read_last)begin
			rd_data_cnt						<= 1'b0;
		end
		else if(axis_dma_read_data[0].ready & axis_dma_read_data[0].valid)begin
			rd_data_cnt						<= rd_data_cnt + 1'b1;
		end
		else begin
			rd_data_cnt						<= rd_data_cnt;
		end		
	end

	assign dma_read_last		= (rd_data_cnt == data_cnt_minus) && axis_dma_read_data[0].ready && axis_dma_read_data[0].valid;




	always@(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_th_en						<= 1'b0;
		end
		else if(dma_read_last)begin
			rd_th_en						<= 1'b0;
		end		
		else if(axis_dma_read_cmd[0].valid & axis_dma_read_cmd[0].ready)begin
			rd_th_en						<= 1'b1;
		end  
		else begin
			rd_th_en						<= rd_th_en;
		end
	end

	
	always@(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_th_sum						<= 32'b0;
		end
		else if(rd_start_r & ~rd_start_rr)begin
			rd_th_sum						<= 32'b0;
		end 
		else if(rd_th_en)begin
			rd_th_sum						<= rd_th_sum + 1'b1;
		end
		else begin
			rd_th_sum						<= rd_th_sum;
		end
	end

	always@(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_lat_en						<= 1'b0;
		end
		else if(axis_dma_read_data[0].valid & axis_dma_read_data[0].ready)begin
			rd_lat_en						<= 1'b0;
		end		
		else if(axis_dma_read_cmd[0].valid & axis_dma_read_cmd[0].ready)begin
			rd_lat_en						<= 1'b1;
		end  
		else begin
			rd_lat_en						<= rd_lat_en;
		end
	end

	
	always@(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_lat_sum						<= 32'b0;
		end
		else if(rd_start_r & ~rd_start_rr)begin
			rd_lat_sum						<= 32'b0;
		end 
		else if(rd_lat_en)begin
			rd_lat_sum						<= rd_lat_sum + 1'b1;
		end
		else begin
			rd_lat_sum						<= rd_lat_sum;
		end
	end

	always @(posedge pcie_clk)begin
		if(~pcie_aresetn)
			error_cnt <= 1'b0;
		else if(((rd_data_cnt + offset) != axis_dma_read_data[0].data[31:0]) && axis_dma_read_data[0].valid && axis_dma_read_data[0].ready)
			error_cnt <= error_cnt + 1'b1;   
		else
			error_cnt <= error_cnt;
	end
	
	always @(posedge pcie_clk)begin
		if(~pcie_aresetn)
			error_index <= 1'b0;
		else if(rd_start_r & ~rd_start_rr)
			error_index <= 1'b0;
		else if(error_flag & ~error_flag_r)
			error_index <= rd_data_cnt;   
		else
			error_index <= error_index;
	end

	always @(posedge pcie_clk)begin
		if(~pcie_aresetn)
			error_flag <= 1'b0;
		else if(rd_start_r & ~rd_start_rr)
			error_flag <= 1'b0; 			
		else if(((rd_data_cnt + offset) != axis_dma_read_data[0].data[31:0]) && axis_dma_read_data[0].valid && axis_dma_read_data[0].ready)
			error_flag <= 1'b1;   
		else
			error_flag <= error_flag;
	end

	always@(posedge pcie_clk)begin
		error_flag_r 	<= error_flag;
	end
	

	assign fpga_status_reg[55] = rd_th_sum;
	assign fpga_status_reg[56] = rd_lat_sum;

	assign fpga_status_reg[57] = error_cnt;
	assign fpga_status_reg[58] = error_index;	


	ila_0 rx (
		.clk(pcie_clk), // input wire clk
	
	
		.probe0(axis_dma_read_cmd[0].valid), // input wire [0:0]  probe0  
		.probe1(axis_dma_read_cmd[0].ready), // input wire [0:0]  probe1 
		.probe2(axis_dma_read_cmd[0].address), // input wire [63:0]  probe2 
		.probe3(axis_dma_read_cmd[0].length), // input wire [31:0]  probe3 
		.probe4(axis_dma_read_data[0].valid), // input wire [0:0]  probe4 
		.probe5(axis_dma_read_data[0].ready), // input wire [0:0]  probe5 
		.probe6(axis_dma_read_data[0].last), // input wire [0:0]  probe6 
		.probe7(axis_dma_read_data[0].data), // input wire [511:0]  probe7
		.probe8(rd_th_sum), // input wire [31:0]  probe8 
		.probe9(rd_lat_sum) // input wire [31:0]  probe9		
	);

//	ila_0 tx (
//		.clk(pcie_clk), // input wire clk
	
	
//		.probe0(axis_dma_write_cmd[0].valid), // input wire [0:0]  probe0  
//		.probe1(axis_dma_write_cmd[0].ready), // input wire [0:0]  probe1 
//		.probe2(axis_dma_write_cmd[0].address), // input wire [63:0]  probe2 
//		.probe3(axis_dma_write_cmd[0].length), // input wire [31:0]  probe3 
//		.probe4(axis_dma_write_data[0].valid), // input wire [0:0]  probe4 
//		.probe5(axis_dma_write_data[0].ready), // input wire [0:0]  probe5 
//		.probe6(axis_dma_write_data[0].last), // input wire [0:0]  probe6 
//		.probe7(axis_dma_write_data[0].data), // input wire [511:0]  probe7
//		.probe8(wr_th_sum), // input wire [31:0]  probe8 
//		.probe9(error_cnt) // input wire [31:0]  probe9		
//	);






/////////////////////////dma debug///////////

//test dma speed
// reg [31:0]		error_cnt;
// reg [31:0]		error_index;


//   wire              user_clk;
//   reg            	read_start;
//   reg				read_start_d;
//   reg            	write_start;
//   reg				write_start_d;
//   reg [31:0]		axis_dma_write_data_cnt;
//   reg [31:0]		axis_dma_write_data_length;
//   reg [31:0]		axis_dma_read_data_cnt;
//   reg [31:0]		axis_dma_read_data_length;
//   reg [31:0]		ops;

//   reg [31:0]		read_cnt;
//   reg [31:0]		write_cnt;
//   reg 				read_cnt_en;
//   reg 				write_cnt_en;
//   reg [31:0]		wr_op_cnt;
//   reg [31:0]		rd_op_cnt;
//   reg [31:0]		wr_op_data_cnt;
//   reg [31:0]		rd_op_data_cnt;  

// assign user_clk = pcie_clk;
// assign user_aresetn = pcie_aresetn;

// always @(posedge user_clk)begin
// 	read_start_d <= read_start;
// 	write_start_d <= write_start;
// end




// always @(posedge user_clk)begin
// 	if(~pcie_aresetn)
// 		axis_dma_read_cmd.valid <= 1'b0;
// 	else if(read_start && ~read_start_d)
// 		axis_dma_read_cmd.valid <= 1'b1;
// 	else if(axis_dma_read_cmd.valid && axis_dma_read_cmd.ready)
// 		axis_dma_read_cmd.valid <= 1'b0;
// 	else 
// 		axis_dma_read_cmd.valid <= axis_dma_read_cmd.valid;
// end

// always @(posedge user_clk)begin
// 	if(~pcie_aresetn)
// 		axis_dma_write_cmd.valid <= 1'b0;
// 	else if(write_start && ~write_start_d)
// 		axis_dma_write_cmd.valid <= 1'b1;
// 	else if(axis_dma_write_cmd.valid && axis_dma_write_cmd.ready)
// 		axis_dma_write_cmd.valid <= 1'b0;
// 	else 
// 		axis_dma_write_cmd.valid <= axis_dma_write_cmd.valid;
// end

// always @(posedge user_clk)begin
// 	if(~pcie_aresetn)
// 		axis_dma_write_data_cnt <= 1'b0;
// 	else if(axis_dma_write_data.last)
// 		axis_dma_write_data_cnt <= 1'b0;
// 	else if(axis_dma_write_data.valid && axis_dma_write_data.ready)
// 		axis_dma_write_data_cnt <= axis_dma_write_data_cnt + 1;    
// 	else
// 		axis_dma_write_data_cnt <= axis_dma_write_data_cnt;
// end

// always @(posedge user_clk)begin
// 	axis_dma_write_data_length <= (axis_dma_write_cmd.length>>6) - 1;
// end


// always @(posedge user_clk)begin
// 	if(~pcie_aresetn)
// 		axis_dma_read_data_cnt <= 1'b0;
// 	else if((axis_dma_read_data_cnt == axis_dma_read_data_length) && axis_dma_read_data.valid && axis_dma_read_data.ready)
// 		axis_dma_read_data_cnt <= 1'b0;
// 	else if(axis_dma_read_data.valid && axis_dma_read_data.ready)
// 		axis_dma_read_data_cnt <= axis_dma_read_data_cnt + 1;    
// 	else
// 		axis_dma_read_data_cnt <= axis_dma_read_data_cnt;
// end

// always @(posedge user_clk)begin
// 	axis_dma_read_data_length <= (axis_dma_read_cmd.length>>6) - 1;
// end

// assign axis_dma_read_data.ready = 1'b1;
// assign axis_dma_write_data.valid = 1'b1;
// assign axis_dma_write_data.keep = 64'hffff_ffff_ffff_ffff;
// assign axis_dma_write_data.data = axis_dma_write_data_cnt;
// assign axis_dma_write_data.last = axis_dma_write_data.valid && axis_dma_write_data.ready && (axis_dma_write_data_cnt == axis_dma_write_data_length);


// //vio_0 your_instance_name (
// //  .clk(user_clk),                // input wire clk
// //  .probe_out0(axis_dma_read_cmd.address),  // output wire [63 : 0] probe_out0
// //  .probe_out1(axis_dma_read_cmd.length),  // output wire [31 : 0] probe_out1
// //  .probe_out2(read_start),  // output wire [0 : 0] probe_out2
// //  .probe_out3(axis_dma_write_cmd.address),  // output wire [63 : 0] probe_out3
// //  .probe_out4(axis_dma_write_cmd.length),  // output wire [31 : 0] probe_out4
// //  .probe_out5(write_start)  // output wire [0 : 0] probe_out5
// //);


// always @(posedge user_clk)begin
// 	if(~pcie_aresetn)
// 		error_cnt <= 1'b0;
// 	else if((axis_dma_read_data_cnt != axis_dma_read_data.data[479:448]) && axis_dma_read_data.valid && axis_dma_read_data.ready)
// 		error_cnt <= error_cnt + 1'b1;   
// 	else
// 		error_cnt <= error_cnt;
// end

// always @(posedge user_clk)begin
// 	if(~pcie_aresetn)
// 		error_index <= 1'b0;
// 	else if((axis_dma_read_data_cnt != axis_dma_read_data.data[479:448]) && axis_dma_read_data.valid && axis_dma_read_data.ready)
// 		error_index <= axis_dma_read_data_cnt;   
// 	else
// 		error_index <= error_index;
// end

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
