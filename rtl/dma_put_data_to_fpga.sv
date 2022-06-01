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

module dma_put_data_to_fpga#(
    parameter   PAGE_SIZE = 2*1024*1024,
    parameter   PAGE_NUM  = 109,
    parameter   CTRL_NUM  = 1024 
)( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,

    //DMA Commands
   	axis_mem_cmd.master         axis_dma_read_cmd,
    //DMA Data streams      
	axi_stream.slave            axis_dma_read_data,
	
    // memory cmd streams
    axis_mem_cmd.master    		m_axis_mem_write_cmd,
    // memory sts streams
    axis_mem_status.slave     	s_axis_mem_write_sts,
    // memory data streams
    axi_stream.master   		m_axis_mem_write_data,

	//control reg
	axis_meta.slave				s_axis_put_data_cmd,
	input wire[15:0][31:0]		control_reg,
	output wire[1:0][31:0]		status_reg

	
	);
	
	axi_stream                  axis_mem_write_data();
   


///////////////////////dma_ex debug//////////

	reg [63:0]            					dma_base_addr;                      
	reg [63:0]            					dma_total_length;

	localparam [3:0]		IDLE 			= 4'h1,
							START			= 4'h2,
							READ_CMD		= 4'h3,
							READ_DATA		= 4'h4,
							JUDGE			= 4'h7,
							END         	= 4'h8;	

	reg [3:0]								state;							

	reg 									rd_start_r,rd_start_rr;
	reg [31:0]								data_cnt,rd_data_cnt;
	reg [31:0]								data_cnt_minus;

    reg [63:0]                              current_addr;
	reg [31:0]                              current_length;
	reg [63:0]								current_mem_addr;
	reg 									ddr_flag;




	reg 									wr_th_en;
	reg [31:0]								wr_th_sum;	



	always @(posedge clk)begin
		dma_base_addr						<= {control_reg[1],control_reg[0]};
		dma_total_length					<= {control_reg[3],control_reg[2]};
	end

    always @(posedge clk)begin
		data_cnt_minus						<= (current_length>>>6)-1;
	end


	//////////////cmd buffer ///////////
	reg 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_almostfull;	
	wire 									fifo_cmd_empty;	
	wire [159:0]								fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;	

	assign fifo_cmd_wr_en					= s_axis_put_data_cmd.ready & s_axis_put_data_cmd.valid;

	assign s_axis_put_data_cmd.ready 		= ~fifo_cmd_almostfull;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 160 ), //64 
		.FIFO_DEPTH_BITS ( 10 )  //determine the size of 16  13
	) inst_a_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_cmd_wr_en     ), //or one cycle later...
	.din        (s_axis_put_data_cmd.data ),
	.almostfull (fifo_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_cmd_rd_en     ),
	.dout       (fifo_cmd_rd_data   ),
	.valid      (fifo_cmd_rd_valid	),
	.empty      (fifo_cmd_empty     ),
	.count      (   )
	);

////////////////////////////////////////////////////////
	reg 									mem_write_cmd_valid;	

	assign	axis_dma_read_cmd.address	    = dma_base_addr + current_addr;
	assign	axis_dma_read_cmd.length	    = current_length; 
	assign 	axis_dma_read_cmd.valid			= (state == READ_CMD); 
	
	assign 	m_axis_mem_write_cmd.address	= current_mem_addr;
	assign  m_axis_mem_write_cmd.length	= current_length;	
	assign 	m_axis_mem_write_cmd.valid	= mem_write_cmd_valid;

	assign s_axis_mem_write_sts.ready 			= 1;

	always@(posedge clk)begin
		if(~rstn)begin
			mem_write_cmd_valid				<= 1'b0;
		end
		else if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
			mem_write_cmd_valid				<= 1'b1;
		end
		else if(m_axis_mem_write_cmd.ready & m_axis_mem_write_cmd.valid)begin
			mem_write_cmd_valid				<= 1'b0;
		end		
		else begin
			mem_write_cmd_valid				<= mem_write_cmd_valid;
		end
	end
	
	axi_stream 								axis_dma_data();

	assign axis_dma_data.valid = axis_dma_read_data.valid;
	assign axis_dma_read_data.ready = axis_dma_data.ready;
	assign axis_dma_data.data = axis_dma_read_data.data;
	assign axis_dma_data.last = (data_cnt == data_cnt_minus) && axis_dma_read_data.ready && axis_dma_read_data.valid;
	assign axis_dma_data.keep = axis_dma_read_data.keep;

	// register_slice_wrapper read_data_slice(
	// 	.aclk								(clk),
	// 	.aresetn							(rstn),
	// 	.s_axis								(axis_dma_data),
	// 	.m_axis								(m_axis_tx_data)
	// );


	axis_data_fifo_512_d1024 read_data_slice_fifo (
		.s_axis_aresetn(rstn),          // input wire s_axis_aresetn
		.s_axis_aclk(clk),                // input wire s_axis_aclk
		.s_axis_tvalid(axis_dma_data.valid),            // input wire s_axis_tvalid
		.s_axis_tready(axis_dma_data.ready),            // output wire s_axis_tready
		.s_axis_tdata(axis_dma_data.data),              // input wire [511 : 0] s_axis_tdata
		.s_axis_tkeep(axis_dma_data.keep),              // input wire [63 : 0] s_axis_tkeep
		.s_axis_tlast(axis_dma_data.last),              // input wire s_axis_tlast
		.m_axis_tvalid(axis_mem_write_data.valid),            // output wire m_axis_tvalid
		.m_axis_tready(axis_mem_write_data.ready),            // input wire m_axis_tready
		.m_axis_tdata(axis_mem_write_data.data),              // output wire [511 : 0] m_axis_tdata
		.m_axis_tkeep(axis_mem_write_data.keep),              // output wire [63 : 0] m_axis_tkeep
		.m_axis_tlast(axis_mem_write_data.last),              // output wire m_axis_tlast
		.axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
		.axis_rd_data_count()  // output wire [31 : 0] axis_rd_data_count
	);

	assign m_axis_mem_write_data.valid = axis_mem_write_data.valid;
	assign axis_mem_write_data.ready = m_axis_mem_write_data.ready;
	assign m_axis_mem_write_data.data = axis_mem_write_data.data;
	assign m_axis_mem_write_data.keep = axis_mem_write_data.keep;
	assign m_axis_mem_write_data.last = axis_mem_write_data.last;	


	always @(posedge clk)begin
		if(~rstn)begin
			data_cnt 						<= 1'b0;
		end
		else if(axis_dma_data.last)begin
			data_cnt						<= 1'b0;
		end
		else if (axis_dma_read_data.ready & axis_dma_read_data.valid)begin
			data_cnt						<= data_cnt + 1'b1;
		end
		else begin
			data_cnt						<= data_cnt;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
			state						<= IDLE;
		end
		else begin
			fifo_cmd_rd_en				<= 1'b0;
			case(state)				
				IDLE:begin
					if(~fifo_cmd_empty)begin
						fifo_cmd_rd_en	<= 1'b1;
						state			<= START;
					end
					else begin
						state			<= IDLE;
					end
                end
				START:begin
					if(fifo_cmd_rd_valid)begin
						state           	<= READ_CMD;
						current_addr		<= fifo_cmd_rd_data[95:32];
						current_length		<= fifo_cmd_rd_data[31:0];
						current_mem_addr	<= fifo_cmd_rd_data[159:96];
						ddr_flag			<= fifo_cmd_rd_data[159];
					end
					else begin
						state			<= START;
					end
                end
				READ_CMD:begin
					if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
						state			<= READ_DATA;
					end
					else begin
						state			<= READ_CMD;
					end
				end
				READ_DATA:begin
					if(axis_mem_write_data.last)begin
						state			<= END;
					end	
					else begin
						state			<= READ_DATA;
					end
                end
				END:begin
					state			<= IDLE;
				end
			endcase
		end
	end

	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		wr_th_en						<= 1'b0;
	// 	end  
	// 	else if(state == END)begin
	// 		wr_th_en						<= 1'b0;
	// 	end
	// 	else if(axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
	// 		wr_th_en						<= 1'b1;
	// 	end		
	// 	else begin
	// 		wr_th_en						<= wr_th_en;
	// 	end
	// end

	
	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		wr_th_sum						<= 32'b0;
	// 	end
	// 	else if(wr_start_r & ~wr_start_rr)begin
	// 		wr_th_sum						<= 32'b0;
	// 	end 
	// 	else if(wr_th_en)begin
	// 		wr_th_sum						<= wr_th_sum + 1'b1;
	// 	end
	// 	else begin
	// 		wr_th_sum						<= wr_th_sum;
	// 	end
	// end



	// ila_3 rx (
	// 	.clk(clk), // input wire clk
	
	
	// 	.probe0(axis_dma_read_cmd.valid), // input wire [0:0]  probe0  
	// 	.probe1(axis_dma_read_cmd.ready), // input wire [0:0]  probe1 
	// 	.probe2(axis_dma_read_cmd.address), // input wire [63:0]  probe2 
	// 	.probe3(axis_dma_read_cmd.length), // input wire [31:0]  probe3 
	// 	.probe4(axis_dma_read_data.valid), // input wire [0:0]  probe4 
	// 	.probe5(axis_dma_read_data.ready), // input wire [0:0]  probe5 
	// 	.probe6(axis_dma_read_data.last), // input wire [0:0]  probe6 
	// 	.probe7(axis_dma_read_data.data[31:0]), // input wire [511:0]  probe7
	// 	.probe8(axis_dma_read_data.keep),
	// 	.probe9(state) // input wire [31:0]  probe8 
	// 	// .probe9(error_cnt) // input wire [31:0]  probe9		
	// );

	// ila_0 tx (
	// 	.clk(clk), // input wire clk
	
	
	// 	.probe0(axis_dma_write_cmd.valid), // input wire [0:0]  probe0  
	// 	.probe1(axis_dma_write_cmd.ready), // input wire [0:0]  probe1 
	// 	.probe2(axis_dma_write_cmd.address), // input wire [63:0]  probe2 
	// 	.probe3(axis_dma_write_cmd.length), // input wire [31:0]  probe3 
	// 	.probe4(axis_dma_write_data.valid), // input wire [0:0]  probe4 
	// 	.probe5(axis_dma_write_data.ready), // input wire [0:0]  probe5 
	// 	.probe6(axis_dma_write_data.last), // input wire [0:0]  probe6 
	// 	.probe7(axis_dma_write_data.data), // input wire [511:0]  probe7
	// 	.probe8(wr_th_sum), // input wire [31:0]  probe8 
	// 	.probe9({state,gpu_write_cnt[27:0]}) // input wire [31:0]  probe9		
	// );








endmodule
