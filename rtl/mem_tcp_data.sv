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

module mem_tcp_data#(
    parameter   SESSION_SIZE = 32*1024*1024,
    parameter   MAX_SESSION_NUM  = 16,
    parameter   TIME_OUT  =  250000
)( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,

    //DMA Commands
   	axis_mem_cmd.master         axis_ddr_write_cmd,
    //DMA Data streams      
	axi_stream.master           axis_ddr_write_data,
	
	//tcp recv
    axis_meta.slave    			s_axis_notifications,
    axis_meta.master     		m_axis_read_package,
    
    axis_meta.slave    			s_axis_rx_metadata,
    axi_stream.slave   			s_axis_rx_data,

	//control reg
	// input wire[15:0][31:0]		control_reg,
	// output wire[1:0][31:0]		status_reg

	);
	
///////////////////////dma_ex debug//////////

	localparam [3:0]		IDLE 			= 4'h1,
							START			= 4'h2,
							WRITE_CMD		= 4'h3,
							WRITE_DATA		= 4'h4,
							WRITE_CTRL_CMD	= 4'h5,
							WRITE_CTRL_DATA	= 4'h6,
							JUDGE			= 4'h7,
							END         	= 4'h8;	

	reg [3:0]								state;							

    tcp_notification                        session_info[MAX_SESSION_NUM]();
	reg [31:0]								start_addr[MAX_SESSION_NUM-1:0];
	reg [31:0]								end_addr[MAX_SESSION_NUM-1:0];
	reg [31:0]								time_cnt;
	reg 									pingpang_flag;	
	reg 									data_flag[MAX_SESSION_NUM-1:0];							

	reg [31:0]								data_cnt,rd_data_cnt;
	reg [31:0]								data_cnt_minus;

    reg [63:0]                              current_addr,current_ctrl_addr,offset_addr;
	reg [15:0]                              length;
	reg [15:0]								session_id;
	reg [31:0]								des_ip_addr;
	reg [15:0]								des_port;
	reg [7:0]								session_close_flag;
    reg [31:0]                              read_count,write_count;
	reg 									judge,judge_r;


	reg 									wr_th_en;
	reg [31:0]								wr_th_sum;				

	always@(posedge clk)begin
		if(~rstn)begin
			time_cnt						<= 0;
		end
		else if(time_cnt > TIME_OUT)begin
			time_cnt						<= 0;
		end
		else begin
			time_cnt						<= time_cnt + 1'b1;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			pingpang_flag					<= 0;
		end
		else if(time_cnt > TIME_OUT)begin
			pingpang_flag					<= ~pingpang_flag;
		end
		else begin
			pingpang_flag					<= pingpang_flag;
		end
	end	

genvar i;
generate
	for(i = 0; i < MAX_SESSION_NUM; i = i + 1) begin
		always@(posedge clk)begin
			start_addr[i]					<= (2 * SESSION_SIZE + pingpang_flag) * i ;
		end

		always@(posedge clk)begin
			if(~rstn)begin
				end_addr[i]					<= (2 * SESSION_SIZE + pingpang_flag) * i ;
			end
			else if((session_id == i) && (axis_ddr_write_cmd.ready & axis_ddr_write_cmd.valid))begin
				end_addr[i]					<= end_addr[i] + length;
			end
			else begin
				end_addr[i]					<= end_addr[i];
			end
		end

		always@(posedge clk)begin
			if(~rstn)begin
				data_flag[i]				<= 0 ;
			end
			else if((session_id == i) && (axis_ddr_write_cmd.ready & axis_ddr_write_cmd.valid))begin
				data_flag[i]				<= 1;
			end
			else begin
				data_flag[i]				<= data_flag[i];
			end
		end


	end

endgenerate


	//////////////notifications buffer ///////////
	wire 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_empty;	
	wire 									fifo_cmd_almostfull;		
	wire [87:0]								fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;			

	assign s_axis_notifications.ready 		= ~fifo_cmd_almostfull;

	assign fifo_cmd_wr_en					= s_axis_notifications.ready && s_axis_notifications.valid;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 88 ), //64 
		.FIFO_DEPTH_BITS ( 10 )  //determine the size of 16  13
	) inst_a_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_cmd_wr_en     ), //or one cycle later...
	.din        (s_axis_notifications.data ),
	.almostfull (fifo_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_cmd_rd_en     ),
	.dout       (fifo_cmd_rd_data   ),
	.valid      (fifo_cmd_rd_valid	),
	.empty      (fifo_cmd_empty     ),
	.count      (   )
	);

////////////////////////////////////////////////////////	
////////////////////////////////////////////////////////
	reg 									read_package_valid;	

	assign	axis_ddr_write_cmd.address	    = end_addr[session_id[3:0]];
	assign	axis_ddr_write_cmd.length	    = length; 
	assign 	axis_ddr_write_cmd.valid		= (state == WRITE_CMD) ; 
	
	assign 	m_axis_read_package.data		= {length,session_id};
	assign 	m_axis_read_package.valid		= read_package_valid;

	always@(posedge clk)begin
		if(~rstn)begin
			read_package_valid				<= 1'b0;
		end
		else if((axis_ddr_write_cmd.ready & axis_ddr_write_cmd.valid) && (state == WRITE_CMD))begin
			read_package_valid				<= 1'b1;
		end
		else if(m_axis_read_package.ready & m_axis_read_package.valid)begin
			read_package_valid				<= 1'b0;
		end		
		else begin
			read_package_valid				<= read_package_valid;
		end
	end
	
	axi_stream 								axis_dma_data();


	assign axis_ddr_write_data.valid	= axis_dma_data.valid;
	assign axis_ddr_write_data.keep		= 64'hffff_ffff_ffff_ffff;
	assign axis_ddr_write_data.last		= axis_dma_data.last;
	assign axis_ddr_write_data.data		= axis_dma_data.data;
	assign axis_dma_data.ready 			= axis_ddr_write_data.ready;



	register_slice_wrapper write_data_slice(
		.aclk								(clk),
		.aresetn							(rstn),
		.s_axis								(s_axis_rx_data),
		.m_axis								(axis_dma_data)
	);


	// always @(posedge clk)begin
	// 	if(~rstn)begin
	// 		data_cnt 						<= 1'b0;
	// 	end
	// 	else if(axis_ddr_write_data.last)begin
	// 		data_cnt						<= 1'b0;
	// 	end
	// 	else if((state == WRITE_DATA) && (axis_ddr_write_data.ready & axis_ddr_write_data.valid))begin
	// 		data_cnt						<= data_cnt + 1'b1;
	// 	end
	// 	else begin
	// 		data_cnt						<= data_cnt;
	// 	end		
	// end

	always @(posedge clk)begin
		if(~rstn)begin
			write_count 						<= 1'b0;
		end
		else if( (state == WRITE_CTRL_DATA) && axis_ddr_write_data.ready & axis_ddr_write_data.valid)begin
			write_count							<= write_count + 1'b1;
		end
		else begin
			write_count							<= write_count;
		end		
	end
	
	assign status_reg[0] = write_count;


	always @(posedge clk)begin
		if(~rstn)begin
			state						<= IDLE;
		end
		else begin
			fifo_cmd_rd_en					<= 1'b0;
			case(state)				
				IDLE:begin
					if(~fifo_cmd_empty)begin
						fifo_cmd_rd_en		<= 1'b1;
						state			<= START;
					end
					else begin
						state			<= IDLE;
					end
                end
                START:begin
					if(fifo_cmd_rd_valid)begin
						state           	<= SAVE_INFO;
						length				<= fifo_cmd_rd_data[31:16];
						session_id	        <= fifo_cmd_rd_data[15:0];
						des_ip_addr			<= fifo_cmd_rd_data[63:32];
						des_port			<= fifo_cmd_rd_data[79:64];
						session_close_flag	<= fifo_cmd_rd_data[87:80];
					end
					else begin
						state			<= START;
					end
				end
				SAVE_INFO:begin
					session_info[session_id[3:0]].session	<= session_id;
					session_info[session_id[3:0]].length	<= length;
					session_info[session_id[3:0]].ipaddr	<= des_ip_addr;
					session_info[session_id[3:0]].port		<= des_port;
					session_info[session_id[3:0]].closeflag	<= session_close_flag;
					state									<= WRITE_CMD;
				end
				WRITE_CMD:begin
					if(axis_ddr_write_cmd.ready & axis_ddr_write_cmd.valid)begin
						state			<= WRITE_DATA;
					end
					else begin
						state			<= WRITE_CMD;
					end
				end
				WRITE_DATA:begin
					if(axis_ddr_write_data.last)begin
						state			<= END;
					end	
					else begin
						state			<= WRITE_DATA;
					end
                end
				END:begin
					state			<= IDLE;
				end
			endcase
		end
	end    







endmodule
