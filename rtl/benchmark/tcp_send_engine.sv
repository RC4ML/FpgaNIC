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

module tcp_send_engine( 
    input wire                          clk,
    input wire                          rstn,
    
    axis_meta.master                    m_axis_tx_metadata,
    axi_stream.master                   m_axis_tx_data,
    axis_meta.slave                     s_axis_tx_status,

	axis_meta.slave                     s_axis_rx_metadata,
    
    input wire[15:0][31:0]              control_reg,
    output wire[7:0][31:0]             status_reg

    );
    

///////////////////////dma_ex debug//////////

    localparam [3:0]		IDLE 			= 4'h0,
                            WRITE_CMD		= 4'h1,
                            JUDGE		    = 4'h2,
							RESP_CMD		= 4'h3,
							RESP_DATA		= 4'h4;

    reg [3:0]								state;							


    reg 									start_r,start_rr,start_rrr,start_rrrr;
    reg [31:0]								data_cnt;
    reg [31:0]								data_cnt_minus;



    reg [15:0]                          session_id,resp_session_id;
	reg [31:0]                          tcp_length,resp_length;
	reg  								server_flag;  //1: server,0: client
	reg [31:0]							offset;
    reg [31:0]                          ops;
    reg [31:0]                          op_nums;
    reg [31:0]                          data_op_nums;

	reg 								write_data_valid;	

always @(posedge clk)begin
    session_id                          <= control_reg[0][15:0];
    tcp_length                          <= control_reg[1];
	ops                                 <= control_reg[2];
	offset								<= control_reg[3];
	start_r							    <= control_reg[7][1];
	server_flag							<= control_reg[8][0];
	start_rr							<= start_r;	
	start_rrr							<= start_rr;		
	start_rrrr							<= start_rrr;		
end


assign m_axis_tx_metadata.valid = (state == WRITE_CMD) ||(state == RESP_CMD);
assign m_axis_tx_metadata.data = (state == WRITE_CMD) ? {tcp_length,session_id} : {resp_length,resp_session_id};

assign s_axis_rx_metadata.ready = (state == IDLE);




always @(posedge clk)begin
	if(server_flag)
		data_cnt_minus						<= (resp_length>>>6)-1;
	else
		data_cnt_minus						<= (tcp_length>>>6)-1;
end

always @(posedge clk)begin
	if(~rstn)begin
		data_cnt 						<= 1'b0;
	end
	else if((data_cnt == data_cnt_minus) & m_axis_tx_data.ready & m_axis_tx_data.valid)begin
		data_cnt						<= 1'b0;
	end
	else if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
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
	else if((data_cnt == data_cnt_minus) & m_axis_tx_data.ready & m_axis_tx_data.valid)begin
		data_op_nums					<= data_op_nums + 1'b1;
	end
	else begin
		data_op_nums					<= data_op_nums;
	end		
end

assign m_axis_tx_data.valid = write_data_valid || (state == RESP_DATA);//1'b1;
assign m_axis_tx_data.data = data_cnt + offset;
assign m_axis_tx_data.keep = 64'hffff_ffff_ffff_ffff;
assign m_axis_tx_data.last = (data_cnt == data_cnt_minus) & m_axis_tx_data.valid & m_axis_tx_data.ready;
assign s_axis_tx_status.ready = 1;

always @(posedge clk)begin
	if(~rstn)begin
		write_data_valid 				<= 1'b0;
    end
	else if(start_rrr & ~start_rrrr & (~server_flag))begin
		write_data_valid				<= 1'b1;
	end
	else if((data_op_nums == (ops-1)) && m_axis_tx_data.last)begin
		write_data_valid				<= 1'b0;
	end    
	else begin
		write_data_valid				<= write_data_valid;
	end		
end


always @(posedge clk)begin
	if(~rstn)begin
		op_nums 						<= 1'b0;
    end
	else if(start_r & ~start_rr)begin
		op_nums						    <= 1'b0;
	end
	else if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
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
				if(start_r & ~start_rr & (~server_flag))begin
					state			    <= WRITE_CMD;
				end
				else if(s_axis_rx_metadata.valid & s_axis_rx_metadata.ready & server_flag)begin
					state			    <= RESP_CMD;
					resp_length			<= s_axis_rx_metadata.data[31:16];
					resp_session_id		<= s_axis_rx_metadata.data[15:0];
				end
				else begin
					state			    <= IDLE;
				end
			end
			RESP_CMD:begin
				if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
					state			    <= RESP_DATA;
				end
				else begin
					state			    <= RESP_CMD;
				end
			end
			RESP_DATA:begin
				if(m_axis_tx_data.last)begin
					state			    <= IDLE;
				end
				else begin
					state			    <= RESP_DATA;
				end
			end
			WRITE_CMD:begin
				if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
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
reg[31:0]                               tcp_word_cnt;
reg[31:0]								tcp_meta_cnt;

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
		tcp_word_cnt 						    <= 1'b0;
    end
	else if(m_axis_tx_data.valid & m_axis_tx_data.ready)begin
		tcp_word_cnt						    <= tcp_word_cnt + 1'b1;
    end    
	else begin
		tcp_word_cnt						    <= tcp_word_cnt;
	end		
end

always @(posedge clk)begin
	if(~rstn)begin
		tcp_meta_cnt 						    <= 1'b0;
    end
	else if(m_axis_tx_metadata.valid & m_axis_tx_metadata.ready)begin
		tcp_meta_cnt						    <= tcp_meta_cnt + 1'b1;
    end    
	else begin
		tcp_meta_cnt						    <= tcp_meta_cnt;
	end		
end

assign status_reg[0] = th_cnt;
assign status_reg[1] = tcp_meta_cnt;
assign status_reg[2] = tcp_word_cnt;

ila_tcp_send ila_tcp_send_inst (
	.clk(clk), // input wire clk


	.probe0(state), // input wire [3:0]  probe0  
	.probe1(s_axis_rx_metadata.valid), // input wire [0:0]  probe1 
	.probe2(s_axis_rx_metadata.ready), // input wire [0:0]  probe2 
	.probe3(resp_length), // input wire [15:0]  probe3 
	.probe4(resp_session_id), // input wire [15:0]  probe4 
	.probe5(m_axis_tx_metadata.valid), // input wire [0:0]  probe5 
	.probe6(m_axis_tx_metadata.ready), // input wire [0:0]  probe6 
	.probe7(m_axis_tx_data.valid), // input wire [0:0]  probe7 
	.probe8(m_axis_tx_data.ready), // input wire [0:0]  probe8 
	.probe9(m_axis_tx_data.last) // input wire [0:0]  probe9
);


endmodule