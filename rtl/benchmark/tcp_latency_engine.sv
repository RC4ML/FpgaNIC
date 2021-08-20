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

module tcp_latency_engine( 
    input wire                          clk,
    input wire                          rstn,
    
    axis_meta.master                    m_axis_tx_metadata,
    axi_stream.master                   m_axis_tx_data,
	axis_meta.slave                     s_axis_tx_status,
	
    axis_meta.slave                     s_axis_notifications,
    axis_meta.master                    m_axis_read_package,
    
    axis_meta.slave                     s_axis_rx_metadata,
    axi_stream.slave                    s_axis_rx_data,

    
    input wire[15:0][31:0]              control_reg,
    output wire[7:0][31:0]              status_reg 

    );
    

///////////////////////send //////////

    localparam [3:0]		IDLE 			= 4'b0001,
                            WRITE_CMD		= 4'b0010,
                            WRITE_DATA		= 4'b0100;

    reg [3:0]								state;							


    reg 									start_r,start_rr,start_rrr,start_rrrr;
    reg [31:0]								data_cnt;



	reg [15:0]                          session_id;
	reg 								tcp_mode;
	reg [31:0]							tcp_length;

	reg 								write_data_valid;	
	reg [31:0]							tx_data_cnt;
	reg [31:0]							rx_data_cnt;
	reg [31:0]							rx_data_minus;
	reg [31:0]							data_cnt_minus;

	reg [31:0]                          read_pkg_data; //{16b'datalength,16b'sessionid}

	reg									read_pkg_valid;


always @(posedge clk)begin
    session_id                          <= control_reg[0][15:0];
    tcp_mode                          	<= control_reg[1][0];
	start_r                             <= control_reg[2][0];
	tcp_length							<= control_reg[3];
	data_cnt_minus						<= (tcp_length>>>6) - 1;
	rx_data_minus						<= (read_pkg_data[31:16]>>>6) - 1;
	start_rr							<= start_r;	
	start_rrr							<= start_rr;		
	start_rrrr							<= start_rrr;		
end


assign m_axis_tx_metadata.valid = (state == WRITE_CMD);
assign m_axis_tx_metadata.data = tcp_mode ? {tcp_length,session_id} : {16'b0,read_pkg_data};

assign s_axis_tx_status.ready = 1;






always@(posedge clk)begin
	if(~rstn)begin
		tx_data_cnt						<= 1'b0;
	end
	else if(m_axis_tx_data.last)begin
		tx_data_cnt						<= 1'b0;
	end
	else if(m_axis_tx_data.valid && m_axis_tx_data.ready)begin
		tx_data_cnt						<= tx_data_cnt + 1'b1;
	end
	else begin
		tx_data_cnt						<= tx_data_cnt;
	end
end



assign m_axis_tx_data.valid = (state == WRITE_DATA);//1'b1;
assign m_axis_tx_data.data = 512'h1234;
assign m_axis_tx_data.keep = 64'hffff_ffff_ffff_ffff;
assign m_axis_tx_data.last = m_axis_tx_data.valid & m_axis_tx_data.ready && (tx_data_cnt == (tcp_mode? data_cnt_minus : rx_data_minus));


//////////////////////////////////recv



always @(posedge clk)begin
	if(~rstn)
		read_pkg_data                  <= 1'b0;
	else if(s_axis_notifications.valid & s_axis_notifications.ready)
		read_pkg_data                  <= s_axis_notifications.data[31:0];
	else 
		read_pkg_data                  <= read_pkg_data;
end

always @(posedge clk)begin
	if(~rstn)
		read_pkg_valid              <= 1'b0;
	else if(s_axis_notifications.valid & s_axis_notifications.ready)
		read_pkg_valid              <= 1'b1;
	else if(m_axis_read_package.valid & m_axis_read_package.ready)
		read_pkg_valid              <= 1'b0;
	else 
		read_pkg_valid              <= read_pkg_valid;
end

always@(posedge clk)begin
	if(~rstn)begin
		rx_data_cnt						<= 1'b0;
	end
	else if(s_axis_rx_data.valid && s_axis_rx_data.ready)begin
		rx_data_cnt						<= rx_data_cnt + 1'b1;
	end
	else begin
		rx_data_cnt						<= rx_data_cnt;
	end
end


assign s_axis_notifications.ready = state == IDLE;
assign m_axis_read_package.valid = read_pkg_valid;
assign m_axis_read_package.data = read_pkg_data;

assign s_axis_rx_data.ready = 1'b1;
assign s_axis_rx_metadata.ready = 1'b1;


///////////////////////////////fsm

always @(posedge clk)begin
	if(~rstn)begin
		state						    <= IDLE;
	end
	else begin
		case(state)
			IDLE:begin
				if((start_r & ~start_rr) && tcp_mode)begin
					state				<= WRITE_CMD;
				end
				else if(~tcp_mode && s_axis_notifications.ready & s_axis_notifications.valid)begin
					state				<= WRITE_CMD;
				end
				else begin
					state			    <= IDLE;
				end
			end
			WRITE_CMD:begin
				if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
					state			    <= WRITE_DATA;
				end
				else begin
					state			    <= WRITE_CMD;
				end
			end
			WRITE_DATA:begin
				if(m_axis_tx_data.last)begin
					state			    <= IDLE;
				end	
				else begin
					state			    <= WRITE_DATA;
				end
			end
		endcase
	end
end










//////////////////////////////////////////////////////

reg[31:0]                               th_cnt;
reg 									th_en;

always @(posedge clk)begin
	if(~rstn)begin
		th_en 						    <= 1'b0;
    end
	else if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
		th_en						    <= 1'b1;
	end
	else if(rx_data_cnt == data_cnt_minus)begin
		th_en						    <= 1'b0;
    end    
	else begin
		th_en						    <= th_en;
	end		
end 

always @(posedge clk)begin
	if(~rstn)begin
		th_cnt 						    <= 1'b0;
    end
	else if(th_en)begin
		th_cnt						    <= th_cnt + 1;
	end   
	else begin
		th_cnt						    <= th_cnt;
	end		
end 

assign status_reg[0] = th_cnt;


ila_5 ila_5_inst (
	.clk(clk), // input wire clk


	.probe0(m_axis_tx_metadata.valid), // input wire [0:0]  probe0  
	.probe1(m_axis_tx_metadata.ready), // input wire [0:0]  probe1 
	.probe2(m_axis_tx_data.valid), // input wire [0:0]  probe2 
	.probe3(m_axis_tx_data.ready), // input wire [0:0]  probe3 
	.probe4(m_axis_tx_data.last), // input wire [0:0]  probe4 
	.probe5(s_axis_notifications.valid), // input wire [0:0]  probe5 
	.probe6(s_axis_notifications.ready), // input wire [0:0]  probe6 
	.probe7(m_axis_read_package.valid), // input wire [0:0]  probe7 
	.probe8(m_axis_read_package.ready), // input wire [0:0]  probe8 
	.probe9(s_axis_rx_data.valid), // input wire [0:0]  probe9 
	.probe10(s_axis_rx_data.ready), // input wire [0:0]  probe10 
	.probe11(s_axis_rx_data.last), // input wire [0:0]  probe11 
	.probe12(state), // input wire [3:0]  probe12 
	.probe13(th_cnt) // input wire [31:0]  probe13
);
endmodule