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

module tcp_recv_engine( 
    input wire                          clk,
    input wire                          rstn,
    
    // axis_meta.slave                     s_axis_notifications,
    // axis_meta.master                    m_axis_read_package,
    
    axis_meta.slave                     s_axis_rx_metadata,
    axi_stream.slave                    s_axis_rx_data,
    
    input wire[15:0][31:0]              control_reg,
    output wire[7:0][31:0]              status_reg

    );
    

///////////////////////dma_ex debug//////////

    localparam [3:0]		IDLE 			= 4'b0001,
                            WRITE_CMD		= 4'b0010,
                            JUDGE		    = 4'b0100;

    reg [3:0]								state;							


    reg 									start_r,start_rr;
    reg [31:0]								data_cnt;
    reg [31:0]								data_cnt_minus;



    reg [31:0]                          read_pkg_data; //{16b'datalength,16b'sessionid}
	reg [31:0]                          tcp_length;
	reg [31:0]							offset;
    reg [31:0]                          ops;
    reg [31:0]                          once_length;
    reg [31:0]                          op_nums;
	reg [31:0]                          data_op_nums;
	
	reg									read_pkg_valid;



	always @(posedge clk)begin
		tcp_length                          <= control_reg[1];
		ops                                 <= control_reg[2];
		offset								<= control_reg[3];		
	end


	// always @(posedge clk)begin
	// 	if(~rstn)
	// 		read_pkg_data                  <= 1'b0;
	// 	else if(s_axis_notifications.valid & s_axis_notifications.ready)
	// 		read_pkg_data                  <= s_axis_notifications.data[31:0];
	// 	else 
	// 		read_pkg_data                  <= read_pkg_data;
	// end
	
	// always @(posedge clk)begin
	// 	if(~rstn)
	// 		read_pkg_valid              <= 1'b0;
	// 	else if(s_axis_notifications.valid & s_axis_notifications.ready)
	// 		read_pkg_valid              <= 1'b1;
	// 	else if(m_axis_read_package.valid & m_axis_read_package.ready)
	// 		read_pkg_valid              <= 1'b0;
	// 	else 
	// 		read_pkg_valid              <= read_pkg_valid;
	// end

	// assign s_axis_notifications.ready = 1'b1;
	// assign m_axis_read_package.valid = read_pkg_valid;
	// assign m_axis_read_package.data = read_pkg_data;
	
	assign s_axis_rx_data.ready = 1'b1;
	assign s_axis_rx_metadata.ready = 1'b1;
	
	always @(posedge clk)begin
		data_cnt_minus						<= (tcp_length>>>6)-1;
	end
	
	always @(posedge clk)begin
		if(~rstn)begin
			data_cnt 						<= 1'b0;
		end
		else if((data_cnt == data_cnt_minus) & s_axis_rx_data.ready & s_axis_rx_data.valid)begin
			data_cnt						<= 1'b0;
		end
		else if(s_axis_rx_data.ready & s_axis_rx_data.valid)begin
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
		else if((data_cnt == data_cnt_minus) & s_axis_rx_data.ready & s_axis_rx_data.valid)begin
			data_op_nums					<= data_op_nums + 1'b1;
		end
		else begin
			data_op_nums					<= data_op_nums;
		end		
	end



//////////////////////////////////////////////////////

reg 									th_start;
reg[31:0]                               th_cnt;
reg[31:0]                               error_cnt;
reg[31:0]                          		error_index;
reg[31:0]                          		tcp_word_counter;


always @(posedge clk)begin
	if(~rstn)begin
		th_start 						    <= 1'b0;
    end
	else if(s_axis_rx_metadata.valid & s_axis_rx_metadata.ready)begin
		th_start						    <= 1'b1;
	end  
	else if(data_op_nums == ops)begin
		th_start							<= 1'b0;
	end
	else begin
		th_start						    <= th_start;
	end		
end 

always @(posedge clk)begin
	if(~rstn)begin
		th_cnt 						    <= 1'b0;
    end
	else if(th_start)begin
		th_cnt						    <= th_cnt + 1;
	end  
	else begin
		th_cnt						    <= th_cnt;
	end		
end 

always @(posedge clk)begin
	if(~rstn)
		error_cnt <= 1'b0;
	else if(((data_cnt+offset) != s_axis_rx_data.data[31:0]) && s_axis_rx_data.valid && s_axis_rx_data.ready)
		error_cnt <= error_cnt + 1'b1;   
	else
		error_cnt <= error_cnt;
end

always @(posedge clk)begin
	if(~rstn)
		error_index <= 1'b0;
	else if(((data_cnt+offset) != s_axis_rx_data.data[31:0]) && s_axis_rx_data.valid && s_axis_rx_data.ready)
		error_index <= data_cnt;   
	else
		error_index <= error_index;
end

always @(posedge clk)begin
	if(~rstn)
		tcp_word_counter <= 1'b0;
	else if(s_axis_rx_data.valid && s_axis_rx_data.ready)
		tcp_word_counter <= tcp_word_counter + 1;   
	else
		tcp_word_counter <= tcp_word_counter;
end

assign status_reg[0] = th_cnt;
assign status_reg[1] = error_cnt;
assign status_reg[2] = error_index;
assign status_reg[3] = tcp_word_counter;

ila_tcp_recv_engine ila_tcp_recv_engine_inst (
	.clk(clk), // input wire clk


	.probe0(0), // input wire [0:0]  probe0  
	.probe1(0), // input wire [0:0]  probe1 
	.probe2(0), // input wire [0:0]  probe2 
	.probe3(0), // input wire [0:0]  probe3 
	.probe4(0), // input wire [31:0]  probe4 
	.probe5(s_axis_rx_data.ready), // input wire [0:0]  probe5 
	.probe6(s_axis_rx_data.valid), // input wire [0:0]  probe6 
	.probe7(s_axis_rx_data.last), // input wire [0:0]  probe7 
	.probe8(s_axis_rx_data.data[31:0]), // input wire [31:0]  probe8 
	.probe9(error_cnt), // input wire [31:0]  probe9 
	.probe10(error_index), // input wire [31:0]  probe10
	.probe11(data_cnt) // input wire [31:0]  probe10
);



endmodule