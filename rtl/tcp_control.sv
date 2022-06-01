`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/26/2020 08:19:38 PM
// Design Name: 
// Module Name: network_module_100g
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
`timescale 1ns / 1ps
`default_nettype none


module tcp_control#(
    parameter       TIME_OUT_CYCLE =    32'hDF84_7580
)
(
    input wire                  clk,
    input wire                  rst,

    input wire[13:0][31:0]     control_reg,
    output reg[7:0][31:0]     status_reg,
    
    //tcp send interface
    axis_meta.master            m_axis_conn_send,     //21
    axis_meta.master            m_axis_ack_to_send,   //ack to tcp send 22
    axis_meta.master            m_axis_ack_to_recv,   //ack to rcv to set buffer id 21
    axis_meta.slave             s_axis_conn_recv,     //22

    //Application interface streams
    axis_meta.master            m_axis_listen_port,
    axis_meta.slave             s_axis_listen_port_status,
   
    axis_meta.master            m_axis_open_connection,
    axis_meta.slave             s_axis_open_status,
    axis_meta.master            m_axis_close_connection 

);

    reg [15:0]                  session_id;
    reg [4:0]                   buffer_id;

    reg [47:0]                  m_axis_open_connection_data;
    reg [15:0]                  m_axis_listen_port_data;
    reg [15:0]                  m_axis_close_connection_data;
    reg [21:0]                  m_axis_ack_to_send_data;

    always@(posedge clk)begin
        m_axis_listen_port_data                 <= control_reg[0][15:0];
        m_axis_open_connection_data             <= {control_reg[3][15:0],control_reg[2]};
        m_axis_close_connection_data            <= control_reg[6][15:0];
        m_axis_ack_to_send_data                 <= {control_reg[9][31],control_reg[9][4:0],control_reg[8][15:0]};        
    end


    assign    m_axis_listen_port.data                 = m_axis_listen_port_data;
    assign    m_axis_open_connection.data             = m_axis_open_connection_data;
    assign    m_axis_close_connection.data            = m_axis_close_connection_data;
    assign    m_axis_ack_to_send.data                 = m_axis_ack_to_send_data;        
    

    assign    m_axis_ack_to_recv.data                 = m_axis_ack_to_recv_data;


    /*tcp listen start*/
    reg                                         listen_start,listen_start_r;
    reg                                         listen_valid;
    reg [31:0]                                  listen_time_out;
    reg                                         listen_time_en;

    always@(posedge clk)begin
        listen_start                            <= control_reg[1][0];
        listen_start_r                          <= listen_start;
    end

    always@(posedge clk)begin
        if(rst)begin
            listen_valid                        <= 1'b0;
        end
        else if(listen_start & ~listen_start_r)begin
            listen_valid                        <= 1'b1;
        end
        else if(m_axis_listen_port.valid & m_axis_listen_port.ready)begin
            listen_valid                        <= 1'b0;
        end    
        else begin
            listen_valid                        <= listen_valid;
        end
    end

    assign m_axis_listen_port.valid = listen_valid;

    always@(posedge clk)begin
        if(rst)begin
            listen_time_en                        <= 1'b0;
        end
        else if(listen_start & ~listen_start_r)begin
            listen_time_en                        <= 1'b1;
        end
        else if(~listen_start & listen_start_r)begin
            listen_time_en                        <= 1'b0;
        end
        else if(listen_time_out == TIME_OUT_CYCLE)begin
            listen_time_en                        <= 1'b0;
        end    
        else begin
            listen_time_en                        <= listen_time_en;
        end
    end

    always@(posedge clk)begin
        if(rst)begin
            listen_time_out                        <= 1'b0;
        end     
        else if(listen_time_en)begin
            listen_time_out                        <= listen_time_out + 1'b1;
        end    
        else begin
            listen_time_out                        <= 1'b0;
        end
    end

    assign s_axis_listen_port_status.ready = 1'b1;

    always@(posedge clk)begin
        if(rst)begin
            status_reg[0]                       <= 1'b0;
        end
        else if(s_axis_listen_port_status.valid & s_axis_listen_port_status.ready)begin
            status_reg[0]                       <= {30'b0,1'b1,s_axis_listen_port_status.data[0]};
        end  
        else if(~listen_start & listen_start_r)begin
            status_reg[0]                       <= 0;
        end  
        else if(listen_time_out == TIME_OUT_CYCLE)begin
            status_reg[0]                       <= 0;
        end         
        else begin
            status_reg[0]                       <= status_reg[0];
        end
    end

    /*tcp connection start*/
    reg                                         conn_start,conn_start_r,conn_start_rr;
    reg                                         conn_valid,conn_send_valid;
    reg [31:0]                                  conn_time_out;
    reg                                         conn_time_en;

    always@(posedge clk)begin
        conn_start                              <= control_reg[5][0];
        conn_start_r                            <= conn_start;
        conn_start_rr                           <= conn_start_r;
    end

    always@(posedge clk)begin
        if(rst)begin
            conn_valid                          <= 1'b0;
        end
        else if(conn_start & ~conn_start_r)begin
            conn_valid                          <= 1'b1;
        end
        else if(m_axis_open_connection.valid & m_axis_open_connection.ready)begin
            conn_valid                          <= 1'b0;
        end    
        else begin
            conn_valid                          <= conn_valid;
        end
    end

    assign m_axis_open_connection.valid = conn_valid;

    //send gpu conn req

    always@(posedge clk)begin
        if(rst)begin
            conn_send_valid                     <= 1'b0;
        end
        else if(s_axis_open_status.valid & s_axis_open_status.ready & s_axis_open_status.data[16])begin
            conn_send_valid                      <= 1'b1;
        end
        else if(m_axis_conn_send.valid & m_axis_conn_send.ready)begin
            conn_send_valid                      <= 1'b0;
        end    
        else begin
            conn_send_valid                      <= conn_send_valid;
        end
    end

    always@(posedge clk)begin
        if(rst)begin
            session_id                          <= 1'b0;
        end
        else if(s_axis_open_status.valid & s_axis_open_status.ready & s_axis_open_status.data[16])begin
            session_id                          <= s_axis_open_status.data[15:0];
        end   
        else begin
            session_id                          <= session_id;
        end
    end 
     
    assign m_axis_conn_send.valid = conn_send_valid;
    assign m_axis_conn_send.data  = {control_reg[4][4:0],session_id};
    //send to send ack req

    reg                                         ack_start,ack_start_r,ack_start_rr;    
    reg                                         ack_to_send_valid,ack_to_recv_valid;
	reg [20:0]									m_axis_ack_to_recv_data;

    always@(posedge clk)begin
        ack_start                               <= control_reg[10][0];
        ack_start_r                             <= ack_start;
        ack_start_rr                            <= ack_start_r;
    end

    always@(posedge clk)begin
        if(rst)begin
            ack_to_send_valid                    <= 1'b0;
        end
        else if(ack_start & ~ack_start_r)begin
            ack_to_send_valid                     <= 1'b1;
        end
        else if(m_axis_ack_to_send.valid & m_axis_ack_to_send.ready)begin
            ack_to_send_valid                     <= 1'b0;
        end    
        else begin
            ack_to_send_valid                     <= ack_to_send_valid;
        end
    end

    assign m_axis_ack_to_send.valid = ack_to_send_valid;

    ///////////send to recv ack req
    always@(posedge clk)begin
        if(rst)begin
            ack_to_recv_valid                    <= 1'b0;
        end
        else if(ack_start_r & ~ack_start_rr)begin
            ack_to_recv_valid                     <= 1'b1;
        end
		else if(m_axis_conn_send.valid & m_axis_conn_send.ready)begin
			ack_to_recv_valid                     <= 1'b1;
		end
        else if(m_axis_ack_to_recv.valid & m_axis_ack_to_recv.ready)begin
            ack_to_recv_valid                     <= 1'b0;
        end    
        else begin
            ack_to_recv_valid                     <= ack_to_recv_valid;
        end
    end

    always@(posedge clk)begin
        if(rst)begin
            m_axis_ack_to_recv_data               <= 1'b0;
        end
        else if(ack_start & ~ack_start_r)begin
            m_axis_ack_to_recv_data               <= {control_reg[9][4:0],control_reg[8][15:0]};
        end
		else if(m_axis_conn_send.valid & m_axis_conn_send.ready)begin
			m_axis_ack_to_recv_data               <= {control_reg[4][4:0],session_id};
		end    
        else begin
            m_axis_ack_to_recv_data               <= m_axis_ack_to_recv_data;
        end
    end



    assign m_axis_ack_to_recv.valid = ack_to_recv_valid;

    ///////////////////

    always@(posedge clk)begin
        if(rst)begin
            conn_time_en                        <= 1'b0;
        end
        else if(conn_start & ~conn_start_r)begin
            conn_time_en                        <= 1'b1;
        end
        else if(~conn_start & conn_start_r)begin
            conn_time_en                        <= 1'b0;
        end
        else if(conn_time_out == TIME_OUT_CYCLE)begin
            conn_time_en                        <= 1'b0;
        end    
        else begin
            conn_time_en                        <= conn_time_en;
        end
    end

    always@(posedge clk)begin
        if(rst)begin
            conn_time_out                        <= 1'b0;
        end     
        else if(conn_time_en)begin
            conn_time_out                        <= conn_time_out + 1'b1;
        end    
        else begin
            conn_time_out                        <= 1'b0;
        end
    end

    assign s_axis_open_status.ready = 1'b1;
    assign s_axis_conn_recv.ready = 1'b1;

    always@(posedge clk)begin
        if(rst)begin
            status_reg[1]                       <= 1'b0;
        end
        else if(s_axis_open_status.valid & s_axis_open_status.ready & (~s_axis_open_status.data[16]))begin
            status_reg[1]                       <= {14'b0,1'b1,s_axis_open_status.data[16:0]};
        end   
        else if(s_axis_conn_recv.valid & s_axis_conn_recv.ready )begin
            status_reg[1]                       <= {14'b0,1'b1,s_axis_conn_recv.data[21],s_axis_conn_recv.data[15:0]};  //	{ctrl_data[16],current_buffer_id[4:0],current_session_id}; //ctrl_data[16]:success
        end
        else if(~conn_start & conn_start_r)begin
            status_reg[1]                       <= 0;
        end  
        else if(conn_time_out == TIME_OUT_CYCLE)begin
            status_reg[1]                       <= 0;
        end           
        else begin
            status_reg[1]                       <= status_reg[1];
        end
    end

    /*tcp close connection*/
    reg                                         conn_close_start,conn_close_start_r;
    reg                                         conn_close_valid;

    always@(posedge clk)begin
        conn_close_start                        <= control_reg[7][0];
        conn_close_start_r                      <= conn_close_start;
    end

    always@(posedge clk)begin
        if(rst)begin
            conn_close_valid                    <= 1'b0;
        end
        else if(conn_close_start & ~conn_close_start_r)begin
            conn_close_valid                    <= 1'b1;
        end
        else if(m_axis_close_connection.valid & m_axis_close_connection.ready)begin
            conn_close_valid                    <= 1'b0;
        end    
        else begin
            conn_close_valid                    <= conn_close_valid;
        end
    end

    assign m_axis_close_connection.valid = conn_close_valid;

    /**/

// ila_tcp_ctrl ila_tcp_ctrl_inst (
// 	.clk(clk), // input wire clk


// 	.probe0(m_axis_conn_send.valid), // input wire [0:0]  probe0  
// 	.probe1(m_axis_conn_send.ready), // input wire [0:0]  probe1 
// 	.probe2(m_axis_conn_send.data), // input wire [15:0]  probe2 
// 	.probe3(m_axis_ack_to_send.valid), // input wire [0:0]  probe3 
// 	 .probe4(m_axis_ack_to_send.ready), // input wire [0:0]  probe4 
// 	 .probe5(m_axis_ack_to_send.data), // input wire [7:0]  probe5 
// 	 .probe6(m_axis_open_connection.valid), // input wire [0:0]  probe6 
// 	 .probe7(m_axis_open_connection.ready), // input wire [0:0]  probe7 
// 	 .probe8(m_axis_open_connection.data), // input wire [47:0]  probe8 
// 	 .probe9(s_axis_open_status.valid), // input wire [0:0]  probe9 
// 	 .probe10(s_axis_open_status.ready), // input wire [0:0]  probe10 
// 	 .probe11(s_axis_open_status.data), // input wire [23:0]  probe11 
// 	 .probe12(m_axis_ack_to_recv.valid), // input wire [0:0]  probe12 
// 	 .probe13(m_axis_ack_to_recv.ready), // input wire [0:0]  probe13 
// 	 .probe14(m_axis_ack_to_recv.data), // input wire [15:0]  probe14
//	 .probe15(conn_start), // input wire [0:0]  probe15 
//	 .probe16(listen_start), // input wire [0:0]  probe16 
//	 .probe17(session_id) // input wire [7:0]  probe17 	
// );



endmodule

`default_nettype wire