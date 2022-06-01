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

module barrier( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,
	
	//tcp send
    axis_meta.master     		m_axis_tx_metadata,
    axi_stream.master    		m_axis_tx_data,
    axis_meta.slave             s_axis_tx_status,

	//tcp recv   
    axis_meta.slave    			s_axis_rx_metadata,
    axi_stream.slave   			s_axis_rx_data,

	//control reg
	input wire[15:0][31:0]		control_reg,
	output wire[7:0][31:0]		status_reg

	
	);
	

	localparam [3:0]		IDLE 			= 4'h0,
                            CAL_SEND_INDEX	= 4'h2,
                            CAL_SESSION		= 4'h3,
                            SEND		    = 4'h4,
                            SEND_DATA		= 4'h5,
                            RECV_DATA	    = 4'h6,
                            END_BARRIER		= 4'h7;



	reg [3:0]								state;	
    reg                                     barrier_start,barrier_start_r;

    reg [7:0]                               epoch_cnt;
    reg [7:0][15:0]                         session_id;
    reg [7:0]                               node_index,node_num,epoch_num,dest_index;

    reg                                     done;


	always @(posedge clk)begin
        session_id[0]                           <= control_reg[0][15:0];
        session_id[1]                           <= control_reg[1][15:0];
        session_id[2]                           <= control_reg[2][15:0];
        session_id[3]                           <= control_reg[3][15:0];
        session_id[4]                           <= control_reg[4][15:0];
        session_id[5]                           <= control_reg[5][15:0];
        session_id[6]                           <= control_reg[6][15:0];
        session_id[7]                           <= control_reg[7][15:0];
		barrier_start							<= control_reg[8][0];
        barrier_start_r                         <= barrier_start;
        node_index                              <= control_reg[9];
        node_num                                <= control_reg[10];
        epoch_num                               <= control_reg[11];
	end


    assign m_axis_tx_metadata.valid = state == SEND;
    assign m_axis_tx_metadata.data = {32'h40,session_id[dest_index]};


    assign m_axis_tx_data.valid = state == SEND_DATA;
    assign m_axis_tx_data.data = {480'h0,32'hffff_ffff};
    assign m_axis_tx_data.keep = 64'hffff_ffff_ffff_ffff;
    assign m_axis_tx_data.last = 1;

    assign s_axis_tx_status.ready = 1;

    assign s_axis_rx_metadata.ready = 1;
    assign s_axis_rx_data.ready = state == RECV_DATA;

	always @(posedge clk)begin
		if(~rstn)begin
			state							<= IDLE;
            epoch_cnt                       <= 0;
            dest_index                      <= 0;
            done                            <= 0;
		end
		else begin
			case(state)				
				IDLE:begin
					if(barrier_start & ~barrier_start_r)begin
                        done                    <= 0;
						state				    <= CAL_SEND_INDEX;
					end
					else begin
						state				    <= IDLE;
					end
				end
                CAL_SEND_INDEX:begin
                    dest_index                  <= node_index + (1<<epoch_cnt);
                    state                       <= CAL_SESSION;
                end
                CAL_SESSION:begin
                    if(dest_index>node_num)begin
                        dest_index              <= dest_index - node_num-1;
                    end
                    else begin
                        dest_index              <= dest_index;
                    end
                    state                       <= SEND;
                end
                SEND:begin
                    if(m_axis_tx_metadata.valid & m_axis_tx_metadata.ready)begin
                        state                   <= SEND_DATA;
                    end
                end
                SEND_DATA:begin
                    if(m_axis_tx_data.valid & m_axis_tx_data.ready)begin
                        state                   <= RECV_DATA;
                    end
                end
                RECV_DATA:begin
                    if(s_axis_rx_data.valid & s_axis_rx_data.ready & (s_axis_rx_data.data[31:0] == 32'hffff_ffff))begin
                        if(epoch_cnt >= (epoch_num-1))begin
                            state               <= END_BARRIER;
                            epoch_cnt           <= 0;
                        end
                        else begin
                            epoch_cnt           <= epoch_cnt + 1;
                            state               <= CAL_SEND_INDEX;
                        end
                    end
                end
                END_BARRIER:begin
                    state                       <= IDLE;
                    done                        <= 1;
                end
			endcase
		end
	end


    assign status_reg[0][0]                     = done;

    reg                                         time_en;
    reg [31:0]                                  time_cnt;

    always@(posedge clk)begin
        if(~rstn)begin
            time_en                             <= 0;
        end
        else if(barrier_start & ~barrier_start_r)begin
            time_en                             <= 1;
        end
        else if(state == END_BARRIER)begin
            time_en                             <= 0;
        end
        else begin
            time_en                             <= time_en;
        end
    end
        
    always@(posedge clk)begin
        if(~rstn)begin
            time_cnt                             <= 0;
        end
        else if(barrier_start & ~barrier_start_r)begin
            time_cnt                             <= 0;
        end
        else if(time_en)begin
            time_cnt                             <= time_cnt + 1;
        end
        else begin
            time_cnt                             <= time_cnt;
        end
    end

    assign status_reg[1]                        = time_cnt;

    ila_barrier ila_barrier_inst (
        .clk(clk), // input wire clk
    
    
        .probe0(state), // input wire [3:0]  probe0  
        .probe1(epoch_cnt), // input wire [7:0]  probe1 
        .probe2(dest_index), // input wire [7:0]  probe2 
        .probe3(time_cnt) // input wire [31:0]  probe3
    );


endmodule
