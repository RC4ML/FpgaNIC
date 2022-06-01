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

module dma_read_engine( 
    input wire                          clk,
    input wire                          rstn,
    
    //DMA Commands
    axis_mem_cmd.master                 m_axis_dma_read_cmd,

    //DMA Data streams      
    axi_stream.slave                    s_axis_dma_read_data, 
    
    input wire[15:0][31:0]              control_reg,
    output wire[15:0][31:0]             status_reg

    );
    

///////////////////////dma_ex debug//////////

    localparam [3:0]		IDLE 			= 4'b0001,
                            WRITE_CMD		= 4'b0010,
                            JUDGE		    = 4'b0100;

    reg [3:0]								state;							


    reg 									start_r,start_rr;
    reg [31:0]								data_cnt;
    reg [31:0]								data_cnt_minus;



    reg [63:0]                          base_addr;
    reg [31:0]                          dma_length;
    reg [31:0]                          ops;
    reg [31:0]                          once_length;
    reg [31:0]                          op_nums;
    reg [31:0]                          data_op_nums;

    reg [63:0]                          c_addr;




assign	m_axis_dma_read_cmd.address	    = c_addr;
assign	m_axis_dma_read_cmd.length	    = once_length; 
assign 	m_axis_dma_read_cmd.valid		= (state == WRITE_CMD); 	

always @(posedge clk)begin
    base_addr                           <= {control_reg[1],control_reg[0]};
    dma_length                          <= control_reg[4];
    ops                                 <= control_reg[5];
    once_length                         <= control_reg[6];
	start_r							    <= control_reg[7][1];
	start_rr							<= start_r;		
end

always @(posedge clk)begin
	data_cnt_minus						<= (once_length>>>6)-1;
end

always @(posedge clk)begin
	if(~rstn)begin
		data_cnt 						<= 1'b0;
	end
	else if((data_cnt == data_cnt_minus) & s_axis_dma_read_data.ready & s_axis_dma_read_data.valid)begin
		data_cnt						<= 1'b0;
	end
	else if(s_axis_dma_read_data.ready & s_axis_dma_read_data.valid)begin
		data_cnt						<= data_cnt + 1'b1;
	end
	else begin
		data_cnt						<= data_cnt;
	end		
end

always @(posedge clk)begin
	if(~rstn)begin
		data_op_nums 					<= 1'b0;
    end
    else if(start_r & ~start_rr)begin
        data_op_nums 					<= 1'b0;
    end
	else if((data_cnt == data_cnt_minus) & s_axis_dma_read_data.ready & s_axis_dma_read_data.valid)begin
		data_op_nums					<= data_op_nums + 1'b1;
	end
	else begin
		data_op_nums					<= data_op_nums;
	end		
end


assign s_axis_dma_read_data.ready		= 1;





always @(posedge clk)begin
	if(~rstn)begin
		c_addr 						    <= 1'b0;
    end
	else if(start_r & ~start_rr)begin
		c_addr						    <= base_addr;
	end
    else if(c_addr > (base_addr + dma_length))begin
        c_addr                          <= base_addr;
    end
	else if(m_axis_dma_read_cmd.ready & m_axis_dma_read_cmd.valid)begin
		c_addr						    <= c_addr + once_length;
    end    
	else begin
		c_addr						    <= c_addr;
	end		
end

always @(posedge clk)begin
	if(~rstn)begin
		op_nums 						<= 1'b0;
    end
	else if(start_r & ~start_rr)begin
		op_nums						    <= 1'b0;
	end
	else if(m_axis_dma_read_cmd.ready & m_axis_dma_read_cmd.valid)begin
		op_nums						    <= op_nums + 1;
    end    
	else begin
		op_nums						    <= op_nums;
	end		
end


always @(posedge clk)begin
	if(~rstn)begin
		state						    <= IDLE;
	end
	else begin
		case(state)
			IDLE:begin
				if(start_r & ~start_rr)begin
					state			    <= WRITE_CMD;
				end
				else begin
					state			    <= IDLE;
				end
			end
			WRITE_CMD:begin
				if(m_axis_dma_read_cmd.ready & m_axis_dma_read_cmd.valid)begin
					state			    <= JUDGE;
				end
				else begin
					state			    <= WRITE_CMD;
				end
			end
			JUDGE:begin
				if(op_nums == ops)begin
					state			    <= IDLE;
				end	
				else begin
					state			    <= WRITE_CMD;
				end
			end
		endcase
	end
end


//////////////////////////////////////////////////////

reg[31:0]                               th_cnt;
reg[31:0]                               lat_cnt;
reg                                     lat_start,lat_end;
reg[9:0][31:0]                          lat_sum;

always @(posedge clk)begin
	if(~rstn)begin
		th_cnt 						    <= 1'b0;
    end
	else if(start_r & ~start_rr)begin
		th_cnt						    <= 1'b0;
	end
	else if(data_op_nums == ops)begin
		th_cnt						    <= th_cnt;
    end    
	else begin
		th_cnt						    <= th_cnt + 1;
	end		
end 



always @(posedge clk)begin
	if(~rstn)begin
		lat_start 						    <= 1'b0;
    end
	else if(m_axis_dma_read_cmd.ready & m_axis_dma_read_cmd.valid)begin
		lat_start						    <= 1'b1;
	end  
	else begin
		lat_start						    <= lat_start;
	end		
end 

always @(posedge clk)begin
	if(~rstn)begin
		lat_end 						    <= 1'b0;
    end
	else if(s_axis_dma_read_data.ready & s_axis_dma_read_data.valid)begin
		lat_end						    <= 1'b1;
    end    
	else begin
		lat_end						    <= lat_end;
	end		
end 

always @(posedge clk)begin
	if(~rstn)begin
		lat_cnt 						<= 1'b0;
    end    
	else if(lat_start & ~lat_end)begin
		lat_cnt						    <= lat_cnt + 1;
    end    
	else begin
		lat_cnt						    <= lat_cnt;
	end		
end 

// always @(posedge clk)begin
// 	if(~rstn)begin
// 		lat_sum 						    <= 1'b0;
//     end
// 	else if(m_axis_dma_read_cmd.ready & m_axis_dma_read_cmd.valid)begin
// 		lat_sum[op_nums]					 <= lat_cnt;
// 	end 
// 	else begin
// 		lat_sum						        <= lat_sum;
// 	end		
// end



assign status_reg[0] = th_cnt;

assign status_reg[1] = lat_cnt;
// assign status_reg[2] = lat_sum[1];
// assign status_reg[3] = lat_sum[2];
// assign status_reg[4] = lat_sum[3];
// assign status_reg[5] = lat_sum[4];
// assign status_reg[6] = lat_sum[5];
// assign status_reg[7] = lat_sum[6];
// assign status_reg[8] = lat_sum[7];
// assign status_reg[9] = lat_sum[8];
// assign status_reg[10] = lat_sum[9];

//ila_dma_benchmark rx (
//.clk(clk), // input wire clk


//.probe0(m_axis_dma_read_cmd.valid), // input wire [0:0]  probe0  
//.probe1(m_axis_dma_read_cmd.ready), // input wire [0:0]  probe1 
//.probe2(m_axis_dma_read_cmd.address), // input wire [63:0]  probe2 
//.probe3(m_axis_dma_read_cmd.length), // input wire [31:0]  probe3 
//.probe4(s_axis_dma_read_data.valid), // input wire [0:0]  probe4 
//.probe5(s_axis_dma_read_data.ready), // input wire [0:0]  probe5 
//.probe6(s_axis_dma_read_data.last), // input wire [0:0]  probe6 
//.probe7(s_axis_dma_read_data.data) // input wire [511:0]  probe7
//);






endmodule