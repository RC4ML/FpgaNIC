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

module dma_data_transfer#(
    parameter   PAGE_SIZE = 2*1024*1024,
    parameter   PAGE_NUM  = 109,
    parameter   CTRL_NUM  = 1024 
)( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,

    //DMA Commands
//    axis_mem_cmd.master         axis_dma_read_cmd,
    axis_mem_cmd.master         axis_dma_write_cmd,

    //DMA Data streams      
    axi_stream.master           axis_dma_write_data,
    // axi_stream.slave            axis_dma_read_data,

    //control reg
    input wire[63:0]            transfer_base_addr,

    input wire[31:0]            transfer_start_page,                      
    input wire[31:0]            transfer_length,
    input wire[31:0]            transfer_offset,
    input wire[31:0]            work_page_size,
    input wire            		transfer_start,
	input wire[31:0]            gpu_read_count,
	output wire[31:0]            gpu_write_count

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

	reg [3:0]								wr_state;							

    reg [63:0]                              start_addr;
	reg 									wr_start_r,wr_start_rr;
	reg 									rd_start_r,rd_start_rr;
	reg [31:0]								data_cnt,rd_data_cnt;
	reg [31:0]								offset;
	reg [31:0]								data_cnt_minus;

    reg [63:0]                              current_addr,current_ctrl_addr;
    reg [31:0]                              current_length,remain_length;
    reg [31:0]                              gpu_write_cnt,gpu_read_cnt;
    reg [31:0]                              work_page_num;
    reg [31:0]                              ctrl_config_num;
	reg 									judge,judge_r;


	reg 									wr_th_en;
	reg [31:0]								wr_th_sum;
	reg 									rd_th_en;
	reg [31:0]								rd_th_sum;
	reg 									rd_lat_en;
	reg [31:0]								rd_lat_sum;			

	reg [31:0]		                    	error_cnt;
	reg [31:0]		                    	error_index;
	reg 									error_flag,error_flag_r;		



	assign	axis_dma_write_cmd.address	    = (wr_state == WRITE_CMD) ? current_addr : current_ctrl_addr;
	assign	axis_dma_write_cmd.length	    = (wr_state == WRITE_CMD) ? current_length : 32'h40; 
	assign 	axis_dma_write_cmd.valid		= (wr_state == WRITE_CMD) || (wr_state == WRITE_CTRL_CMD); 	

	always @(posedge clk)begin
		wr_start_r							<= transfer_start;
		wr_start_rr							<= wr_start_r;
		rd_start_r							<= transfer_start;
		rd_start_rr							<= rd_start_r;	
		judge								<= (wr_state == JUDGE);
		judge_r								<= judge;
	end

    always @(posedge clk)begin
        start_addr                          <= (transfer_start_page * PAGE_SIZE) + transfer_base_addr;
		data_cnt_minus						<= ((wr_state == WRITE_DATA) || (wr_state == WRITE_CMD)) ? ((current_length>>>6)-1) : 0;
        offset								<= transfer_offset;
        gpu_read_cnt                        <= gpu_read_count;
	end

	always @(posedge clk)begin
		if(~rstn)begin
			current_addr 			        <= 1'b0;
		end
		else if(wr_start_r & ~wr_start_rr)begin
			current_addr					<= start_addr;
        end
        else if(work_page_num == work_page_size)begin
            current_addr                    <= start_addr;
        end
		else if((wr_state == WRITE_CTRL_DATA) && axis_dma_write_data.ready & axis_dma_write_data.valid)begin
			current_addr					<= current_addr + PAGE_SIZE;
		end
		else begin
			current_addr					<= current_addr;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
			work_page_num 			        <= 1'b0;
		end
		else if(wr_start_r & ~wr_start_rr)begin
			work_page_num					<= 1'b0;
        end
        else if(work_page_num == work_page_size)begin
            work_page_num                   <= 1'b0;
        end
		else if((wr_state == WRITE_CTRL_DATA) && axis_dma_write_data.ready & axis_dma_write_data.valid)begin
			work_page_num					<= work_page_num + 1'b1;
		end
		else begin
			work_page_num					<= work_page_num;
		end		
	end


	always @(posedge clk)begin
		if(~rstn)begin
			ctrl_config_num 			    <= 1'b0;
		end
		else if(wr_start_r & ~wr_start_rr)begin
			ctrl_config_num					<= 1'b0;
        end
        else if(ctrl_config_num == CTRL_NUM)begin
            ctrl_config_num                 <= 1'b0;
        end
		else if((wr_state == WRITE_CTRL_DATA) && axis_dma_write_data.ready & axis_dma_write_data.valid)begin
			ctrl_config_num					<= ctrl_config_num + 1'b1;
		end
		else begin
			ctrl_config_num					<= ctrl_config_num;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
			current_ctrl_addr 			    <= 1'b0;
		end
		else if(wr_start_r & ~wr_start_rr)begin
			current_ctrl_addr		        <= transfer_base_addr;
        end
        else if(ctrl_config_num == CTRL_NUM)begin
            current_ctrl_addr               <= transfer_base_addr;
        end
		else if((wr_state == WRITE_CTRL_DATA) && axis_dma_write_data.ready & axis_dma_write_data.valid)begin
			current_ctrl_addr			    <= current_ctrl_addr + 32'h40;
		end
		else begin
			current_ctrl_addr				<= current_ctrl_addr;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
            remain_length 			    <= 1'b0;
		end
        else if(wr_start_r & ~wr_start_rr)begin
            remain_length               <= transfer_length;
        end
		else if(~judge & judge_r && (remain_length > PAGE_SIZE))begin
            remain_length			    <= remain_length - PAGE_SIZE;
        end
		else begin
            remain_length				<= remain_length;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
            current_length              <= 1'b0;
		end
		else if((wr_state == START)  && (remain_length > PAGE_SIZE))begin
            current_length              <= PAGE_SIZE;
        end
        else if(wr_state == START)begin
            current_length              <= transfer_length;
        end
		else if(~judge & judge_r && (remain_length > PAGE_SIZE))begin
            current_length              <= PAGE_SIZE;
        end
        else if(~judge & judge_r)begin
            current_length              <= remain_length;            
        end
		else begin
            current_length              <= current_length;
		end		
	end


	always @(posedge clk)begin
		if(~rstn)begin
			data_cnt 						<= 1'b0;
		end
		else if(axis_dma_write_data.last)begin
			data_cnt						<= 1'b0;
		end
		else if((wr_state == WRITE_DATA) && (axis_dma_write_data.ready & axis_dma_write_data.valid))begin
			data_cnt						<= data_cnt + 1'b1;
		end
		else begin
			data_cnt						<= data_cnt;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
			gpu_write_cnt 						<= 1'b0;
		end
		else if( (wr_state == WRITE_CTRL_DATA) && axis_dma_write_data.ready & axis_dma_write_data.valid)begin
			gpu_write_cnt						<= gpu_write_cnt + 1'b1;
		end
		else begin
			gpu_write_cnt						<= gpu_write_cnt;
		end		
	end
	
	assign gpu_write_count = gpu_write_cnt;

	assign axis_dma_write_data.valid	= (wr_state == WRITE_DATA) || (wr_state == WRITE_CTRL_DATA);
	assign axis_dma_write_data.keep		= 64'hffff_ffff_ffff_ffff;
	assign axis_dma_write_data.last		= (data_cnt == data_cnt_minus) && axis_dma_write_data.ready && axis_dma_write_data.valid;
	assign axis_dma_write_data.data		= (wr_state == WRITE_DATA) ? (data_cnt + offset) : {1'b1,415'h0,work_page_num,ctrl_config_num,current_length};

	always @(posedge clk)begin
		if(~rstn)begin
			wr_state						<= IDLE;
		end
		else begin
			case(wr_state)
				IDLE:begin
					if(wr_start_r & ~wr_start_rr)begin
						wr_state			<= START;
					end
					else begin
						wr_state			<= IDLE;
					end
                end
                START:begin
                    wr_state                <= WRITE_CMD;
                end
				WRITE_CMD:begin
					if(axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
						wr_state			<= WRITE_DATA;
					end
					else begin
						wr_state			<= WRITE_CMD;
					end
				end
				WRITE_DATA:begin
					if(axis_dma_write_data.last)begin
						wr_state			<= WRITE_CTRL_CMD;
					end	
					else begin
						wr_state			<= WRITE_DATA;
					end
                end
                WRITE_CTRL_CMD:begin
					if(axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
						wr_state			<= WRITE_CTRL_DATA;
					end
					else begin
						wr_state			<= WRITE_CTRL_CMD;
					end                    
                end
                WRITE_CTRL_DATA:begin
					if(axis_dma_write_data.last)begin
						wr_state			<= JUDGE;
					end	
					else begin
						wr_state			<= WRITE_CTRL_DATA;
					end                    
				end
				JUDGE:begin
					if(remain_length <= PAGE_SIZE)begin
						wr_state			<= END;
					end
					else if(((gpu_write_cnt >= gpu_read_cnt) && ((gpu_read_cnt + work_page_size) > gpu_write_cnt)) || ((gpu_write_cnt < gpu_read_cnt) && (32'hffff_ffff + gpu_write_cnt < gpu_read_cnt + work_page_size)))begin
						wr_state			<= START;
					end
					else begin
						wr_state			<= JUDGE;
					end
				end
				END:begin
					wr_state			<= IDLE;
				end
			endcase
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_th_en						<= 1'b0;
		end  
		else if(wr_state == END)begin
			wr_th_en						<= 1'b0;
		end
		else if(axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
			wr_th_en						<= 1'b1;
		end		
		else begin
			wr_th_en						<= wr_th_en;
		end
	end

	
	always@(posedge clk)begin
		if(~rstn)begin
			wr_th_sum						<= 32'b0;
		end
		else if(wr_start_r & ~wr_start_rr)begin
			wr_th_sum						<= 32'b0;
		end 
		else if(wr_th_en)begin
			wr_th_sum						<= wr_th_sum + 1'b1;
		end
		else begin
			wr_th_sum						<= wr_th_sum;
		end
	end

// 	assign fpga_status_reg[54] = wr_th_sum;
// //////////////////////////////////////////////////////read

// 	reg 									dma_read_cmd_valid;

// 	assign	axis_dma_read_cmd.address	= {fpga_control_reg[33],fpga_control_reg[32]};
// 	assign	axis_dma_read_cmd.length		= fpga_control_reg[34]; 
// 	assign 	axis_dma_read_cmd.valid		= dma_read_cmd_valid; 

// 	assign  axis_dma_read_data[1].ready		= 1;
// 	assign  axis_dma_read_data.ready		= 1;
// 	assign  axis_dma_read_data[2].ready		= 1;
// 	assign  axis_dma_read_data[3].ready		= 1;

// 	always @(posedge clk)begin
// 		if(~rstn)begin
// 			dma_read_cmd_valid				<= 1'b0;
// 		end
// 		else if(rd_start_r & ~rd_start_rr)begin
// 			dma_read_cmd_valid				<= 1'b1;
// 		end
// 		else if(axis_dma_read_cmd.valid & axis_dma_read_cmd.ready)begin
// 			dma_read_cmd_valid				<= 1'b0;
// 		end
// 		else begin
// 			dma_read_cmd_valid				<= dma_read_cmd_valid;
// 		end		
// 	end

// 	always @(posedge clk)begin
// 		if(~rstn)begin
// 			rd_data_cnt 					<= 1'b0;
// 		end
// 		else if(dma_read_last)begin
// 			rd_data_cnt						<= 1'b0;
// 		end
// 		else if(axis_dma_read_data.ready & axis_dma_read_data.valid)begin
// 			rd_data_cnt						<= rd_data_cnt + 1'b1;
// 		end
// 		else begin
// 			rd_data_cnt						<= rd_data_cnt;
// 		end		
// 	end

// 	assign dma_read_last		= (rd_data_cnt == data_cnt_minus) && axis_dma_read_data.ready && axis_dma_read_data.valid;




// 	always@(posedge clk)begin
// 		if(~rstn)begin
// 			rd_th_en						<= 1'b0;
// 		end
// 		else if(dma_read_last)begin
// 			rd_th_en						<= 1'b0;
// 		end		
// 		else if(axis_dma_read_cmd.valid & axis_dma_read_cmd.ready)begin
// 			rd_th_en						<= 1'b1;
// 		end  
// 		else begin
// 			rd_th_en						<= rd_th_en;
// 		end
// 	end

	
// 	always@(posedge clk)begin
// 		if(~rstn)begin
// 			rd_th_sum						<= 32'b0;
// 		end
// 		else if(rd_start_r & ~rd_start_rr)begin
// 			rd_th_sum						<= 32'b0;
// 		end 
// 		else if(rd_th_en)begin
// 			rd_th_sum						<= rd_th_sum + 1'b1;
// 		end
// 		else begin
// 			rd_th_sum						<= rd_th_sum;
// 		end
// 	end

// 	always@(posedge clk)begin
// 		if(~rstn)begin
// 			rd_lat_en						<= 1'b0;
// 		end
// 		else if(axis_dma_read_data.valid & axis_dma_read_data.ready)begin
// 			rd_lat_en						<= 1'b0;
// 		end		
// 		else if(axis_dma_read_cmd.valid & axis_dma_read_cmd.ready)begin
// 			rd_lat_en						<= 1'b1;
// 		end  
// 		else begin
// 			rd_lat_en						<= rd_lat_en;
// 		end
// 	end

	
// 	always@(posedge clk)begin
// 		if(~rstn)begin
// 			rd_lat_sum						<= 32'b0;
// 		end
// 		else if(rd_start_r & ~rd_start_rr)begin
// 			rd_lat_sum						<= 32'b0;
// 		end 
// 		else if(rd_lat_en)begin
// 			rd_lat_sum						<= rd_lat_sum + 1'b1;
// 		end
// 		else begin
// 			rd_lat_sum						<= rd_lat_sum;
// 		end
// 	end

// 	always @(posedge clk)begin
// 		if(~rstn)
// 			error_cnt <= 1'b0;
// 		else if(((rd_data_cnt + offset) != axis_dma_read_data.data[31:0]) && axis_dma_read_data.valid && axis_dma_read_data.ready)
// 			error_cnt <= error_cnt + 1'b1;   
// 		else
// 			error_cnt <= error_cnt;
// 	end
	
// 	always @(posedge clk)begin
// 		if(~rstn)
// 			error_index <= 1'b0;
// 		else if(rd_start_r & ~rd_start_rr)
// 			error_index <= 1'b0;
// 		else if(error_flag & ~error_flag_r)
// 			error_index <= rd_data_cnt;   
// 		else
// 			error_index <= error_index;
// 	end

// 	always @(posedge clk)begin
// 		if(~rstn)
// 			error_flag <= 1'b0;
// 		else if(rd_start_r & ~rd_start_rr)
// 			error_flag <= 1'b0; 			
// 		else if(((rd_data_cnt + offset) != axis_dma_read_data.data[31:0]) && axis_dma_read_data.valid && axis_dma_read_data.ready)
// 			error_flag <= 1'b1;   
// 		else
// 			error_flag <= error_flag;
// 	end

// 	always@(posedge clk)begin
// 		error_flag_r 	<= error_flag;
// 	end
	

// 	assign fpga_status_reg[55] = rd_th_sum;
// 	assign fpga_status_reg[56] = rd_lat_sum;

// 	assign fpga_status_reg[57] = error_cnt;
// 	assign fpga_status_reg[58] = error_index;	


	// ila_0 rx (
	// 	.clk(clk), // input wire clk
	
	
	// 	.probe0(axis_dma_read_cmd.valid), // input wire [0:0]  probe0  
	// 	.probe1(axis_dma_read_cmd.ready), // input wire [0:0]  probe1 
	// 	.probe2(axis_dma_read_cmd.address), // input wire [63:0]  probe2 
	// 	.probe3(axis_dma_read_cmd.length), // input wire [31:0]  probe3 
	// 	.probe4(axis_dma_read_data.valid), // input wire [0:0]  probe4 
	// 	.probe5(axis_dma_read_data.ready), // input wire [0:0]  probe5 
	// 	.probe6(axis_dma_read_data.last), // input wire [0:0]  probe6 
	// 	.probe7(axis_dma_read_data.data), // input wire [511:0]  probe7
	// 	.probe8(rd_th_sum), // input wire [31:0]  probe8 
	// 	.probe9(error_cnt) // input wire [31:0]  probe9		
	// );

	ila_0 tx (
		.clk(clk), // input wire clk
	
	
		.probe0(axis_dma_write_cmd.valid), // input wire [0:0]  probe0  
		.probe1(axis_dma_write_cmd.ready), // input wire [0:0]  probe1 
		.probe2(axis_dma_write_cmd.address), // input wire [63:0]  probe2 
		.probe3(axis_dma_write_cmd.length), // input wire [31:0]  probe3 
		.probe4(axis_dma_write_data.valid), // input wire [0:0]  probe4 
		.probe5(axis_dma_write_data.ready), // input wire [0:0]  probe5 
		.probe6(axis_dma_write_data.last), // input wire [0:0]  probe6 
		.probe7(axis_dma_write_data.data), // input wire [511:0]  probe7
		.probe8(wr_th_sum), // input wire [31:0]  probe8 
		.probe9({wr_state,gpu_write_cnt[27:0]}) // input wire [31:0]  probe9		
	);








endmodule
