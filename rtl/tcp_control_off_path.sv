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


module tcp_control_off_path#(
    parameter       TIME_OUT_CYCLE =    32'h9502_F900
)
(
    input wire                  clk,
    input wire                  rst,

    input wire[13:0][31:0]     control_reg,
    output reg[7:0][31:0]     status_reg,         

    //Application interface streams
    axis_meta.master            m_axis_listen_port,
    axis_meta.slave             s_axis_listen_port_status,
   
    axis_meta.master            m_axis_open_connection,
    axis_meta.slave             s_axis_open_status,
    axis_meta.master            m_axis_close_connection 

);

    


    always@(posedge clk)begin
        m_axis_listen_port.data                 <= control_reg[0][15:0];
        m_axis_open_connection.data             <= {control_reg[3][15:0],control_reg[2]};
        m_axis_close_connection.data            <= control_reg[5][15:0];
    end

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
    reg                                         conn_start,conn_start_r;
    reg                                         conn_valid;
    reg [31:0]                                  conn_time_out;
    reg                                         conn_time_en;

    always@(posedge clk)begin
        conn_start                              <= control_reg[4][0];
        conn_start_r                            <= conn_start;
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

    always@(posedge clk)begin
        if(rst)begin
            status_reg[1]                       <= 1'b0;
        end
        else if(s_axis_open_status.valid & s_axis_open_status.ready)begin
            status_reg[1]                       <= {14'b0,1'b1,s_axis_open_status.data[16:0]};
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
        conn_close_start                        <= control_reg[6][0];
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

//ila_tcp_ctrl ila_tcp_ctrl_inst (
//	.clk(clk), // input wire clk


//	.probe0(m_axis_listen_port.valid), // input wire [0:0]  probe0  
//	.probe1(m_axis_listen_port.ready), // input wire [0:0]  probe1 
//	.probe2(m_axis_listen_port.data), // input wire [15:0]  probe2 
//	.probe3(s_axis_listen_port_status.valid), // input wire [0:0]  probe3 
//	.probe4(s_axis_listen_port_status.ready), // input wire [0:0]  probe4 
//	.probe5(s_axis_listen_port_status.data), // input wire [7:0]  probe5 
//	.probe6(m_axis_open_connection.valid), // input wire [0:0]  probe6 
//	.probe7(m_axis_open_connection.ready), // input wire [0:0]  probe7 
//	.probe8(m_axis_open_connection.data), // input wire [47:0]  probe8 
//	.probe9(s_axis_open_status.valid), // input wire [0:0]  probe9 
//	.probe10(s_axis_open_status.ready), // input wire [0:0]  probe10 
//	.probe11(s_axis_open_status.data), // input wire [23:0]  probe11 
//	.probe12(m_axis_close_connection.valid), // input wire [0:0]  probe12 
//	.probe13(m_axis_close_connection.ready), // input wire [0:0]  probe13 
//	.probe14(m_axis_close_connection.data) // input wire [15:0]  probe14
//);

endmodule

`default_nettype wire