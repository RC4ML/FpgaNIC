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

module dma_write_engine( 
    input wire                          clk,
    input wire                          rstn,
    
    //DMA Commands
    axis_mem_cmd.master          m_axis_dma_write_cmd,

    //DMA Data streams      
    axi_stream.master            m_axis_dma_write_data,
    
    input wire[15:0][31:0]              control_reg,
    output wire[15:0][31:0]             status_reg

    );
    

///////////////////////dma_ex debug//////////

    localparam [3:0]		IDLE 			= 4'b0001,
							WRITE_CMD		= 4'b0010,
							END 			= 4'b1000,
                            JUDGE		    = 4'b0100;

    reg [3:0]								state;							


    reg 									start_r,start_rr,start_rrr,start_rrrr;
    reg [31:0]								data_cnt;
    reg [31:0]								data_cnt_minus;



    reg [63:0]                          base_addr;
    reg [31:0]                          dma_length;
    reg [31:0]                          ops;
    reg [31:0]                          once_length;
    reg [31:0]                          op_nums;
    reg [31:0]                          data_op_nums;
	reg [31:0]							latency_cycle;

    reg [63:0]                          c_addr;




assign	m_axis_dma_write_cmd.address	    = c_addr;
assign	m_axis_dma_write_cmd.length	    = once_length; 
assign 	m_axis_dma_write_cmd.valid		= (state == WRITE_CMD); 	

always @(posedge clk)begin
    base_addr                           <= {control_reg[3],control_reg[2]};
    dma_length                          <= control_reg[4];
    ops                                 <= control_reg[5];
    once_length                         <= control_reg[6];
	start_r							    <= control_reg[7][0];
	latency_cycle						<= control_reg[8];
	start_rr							<= start_r;
	start_rrr							<= start_rr;		
	start_rrrr							<= start_rrr;				
end

always @(posedge clk)begin
	data_cnt_minus						<= (once_length>>>2)-16;
end

always @(posedge clk)begin
	if(~rstn)begin
		data_cnt 						<= 1'b0;
	end
	else if(start_r & ~start_rr)begin
		data_cnt						<= 1'b0;
	end
	else if(m_axis_dma_write_data.ready & m_axis_dma_write_data.valid)begin
		data_cnt						<= data_cnt + 32'd16;
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
	else if((data_cnt == data_cnt_minus) & m_axis_dma_write_data.ready & m_axis_dma_write_data.valid)begin
		data_op_nums					<= data_op_nums + 1'b1;
	end
	else begin
		data_op_nums					<= data_op_nums;
	end		
end

	reg 								write_data_valid;



assign m_axis_dma_write_data.valid		= write_data_valid;
assign m_axis_dma_write_data.keep		= 64'hffff_ffff_ffff_ffff;
assign m_axis_dma_write_data.last		= (data_cnt == data_cnt_minus) && m_axis_dma_write_data.ready && m_axis_dma_write_data.valid;
assign m_axis_dma_write_data.data		= {data_cnt+15,data_cnt+14,data_cnt+13,data_cnt+12,data_cnt+11,data_cnt+10,data_cnt+9,data_cnt+8,data_cnt+7,data_cnt+6,data_cnt+5,data_cnt+4,data_cnt+3,data_cnt+2,data_cnt+1,data_cnt};



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
	else if(m_axis_dma_write_cmd.ready & m_axis_dma_write_cmd.valid)begin
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
	else if(m_axis_dma_write_cmd.ready & m_axis_dma_write_cmd.valid)begin
		op_nums						    <= op_nums + 1;
    end    
	else begin
		op_nums						    <= op_nums;
	end		
end



always @(posedge clk)begin
	if(~rstn)begin
		write_data_valid 				<= 1'b0;
    end
	else if(start_rrr & ~start_rrrr)begin
		write_data_valid				<= 1'b1;
	end
	else if((data_op_nums == (ops-1)) && m_axis_dma_write_data.last)begin
		write_data_valid				<= 1'b0;
	end    
	else begin
		write_data_valid				<= write_data_valid;
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
				if(m_axis_dma_write_cmd.ready & m_axis_dma_write_cmd.valid)begin
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


reg [12:0]						data_count;
reg 							rd_en ;
reg [31:0]						data_cnt_o ;

fifo_32w_8192d latency_fifo (
  .clk(clk),                  // input wire clk
  .srst(~rstn),                // input wire srst
  .din(data_cnt),                  // input wire [31 : 0] din
  .wr_en(1),              // input wire wr_en
  .rd_en(rd_en),              // input wire rd_en
  .dout(data_cnt_o),                // output wire [31 : 0] dout
  .full(),                // output wire full
  .empty(),              // output wire empty
  .data_count(data_count),    // output wire [12 : 0] data_count
  .wr_rst_busy(),  // output wire wr_rst_busy
  .rd_rst_busy()  // output wire rd_rst_busy
);


always @(posedge clk)begin
	if(~rstn)begin
		rd_en 						    <= 1'b0;
    end
	else if(latency_cycle < data_count)begin
		rd_en						    <= 1'b1;
	end  
	else begin
		rd_en						    <= 1'b0;
	end		
end

//////////////////////////////////////////////////////

reg[31:0]                               th_cnt;

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

assign status_reg[0] = th_cnt;
assign status_reg[1] = (data_cnt_o <<< 2);


//ila_dma_benchmark tx (
//.clk(clk), // input wire clk


//.probe0(m_axis_dma_write_cmd.valid), // input wire [0:0]  probe0  
//.probe1(m_axis_dma_write_cmd.ready), // input wire [0:0]  probe1 
//.probe2({data_cnt,data_cnt_o}), // input wire [63:0]  probe2 
//.probe3(data_count), // input wire [31:0]  probe3 
//.probe4(m_axis_dma_write_data.valid), // input wire [0:0]  probe4 
//.probe5(m_axis_dma_write_data.ready), // input wire [0:0]  probe5 
//.probe6(m_axis_dma_write_data.last), // input wire [0:0]  probe6 
//.probe7(m_axis_dma_write_data.data) // input wire [511:0]  probe7

//);



endmodule
