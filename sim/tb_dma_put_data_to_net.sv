`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/21/2020 02:36:51 PM
// Design Name: 
// Module Name: tb_dma_read_data
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

module tb_dma_put_data_to_net(

    );
    reg clk,rstn;

    // DMA Signals
        axis_mem_cmd    axis_dma_read_cmd();
        axi_stream      axis_dma_read_data();
        
        axis_meta #(.WIDTH(48))     axis_tcp_tx_meta();
        axi_stream #(.WIDTH(512))    axis_tcp_tx_data();
        axis_meta #(.WIDTH(64))     axis_tcp_tx_status();
        
        axis_meta #(.WIDTH(112))     s_axis_put_data_to_net();
        axis_meta #(.WIDTH(112))     s_axis_get_data_cmd();
        axis_meta #(.WIDTH(112))     s_axis_put_data_cmd();
        
        
        reg[15:0][31:0]     fpga_control_reg;
        wire[1:0][31:0]     fpga_status_reg; 

        reg                 conn_send_valid;
        reg                 conn_ack_valid;
        reg                 send_read_cnt_valid;  
        reg                 cmd_valid; 
    
        initial begin
            clk = 1'b1;
            rstn = 1'b0;
            fpga_control_reg[0] = 32'h1234_0000;
            fpga_control_reg[1] = 32'h0001_5678;
            fpga_control_reg[2] = 12;
            fpga_control_reg[3] = 3;
            conn_send_valid = 0;
            conn_ack_valid = 0;
            send_read_cnt_valid = 0;
            cmd_valid = 0;
            #500
            rstn = 1'b1;
            #1000
            conn_send_valid = 1;
            #100
            conn_send_valid = 0;
            conn_ack_valid = 1;
            #40
            conn_ack_valid = 0;
            send_read_cnt_valid = 1;
            #40
            send_read_cnt_valid = 0;
            cmd_valid = 1;
            #100
            cmd_valid = 0;
        end
    
        always #5 clk = ~clk;
    
        assign axis_dma_read_cmd.ready = 1'b1;
    
        assign axis_dma_read_data.valid = 1'b1;
        assign axis_dma_read_data.data = 1234;
        assign axis_dma_read_data.keep = 64'hffff_ffff_ffff_ffff;
        assign axis_dma_read_data.last = 1'b0;
    
    
        assign axis_tcp_tx_meta.ready = 1'b1;
        assign axis_tcp_tx_data.ready = 1'b1;


        assign s_axis_put_data_to_net.data = {16'h1,32'h80,32'h40,32'h4000};
        assign s_axis_get_data_cmd.data = {16'h2,32'h80,32'h40,32'h4000};
        assign s_axis_put_data_cmd.data = {16'h3,32'h80,32'h40,32'h4000};
    
        assign s_axis_put_data_to_net.valid = conn_ack_valid;
        assign s_axis_get_data_cmd.valid = conn_send_valid;
        assign s_axis_put_data_cmd.valid = send_read_cnt_valid;       
    
        dma_put_data_to_net dma_put_data_to_net( 
    
            //user clock input
            .clk                        (clk),
            .rstn                       (rstn),
        
            //DMA Commands
            .axis_dma_read_cmd          (axis_dma_read_cmd),
            //DMA Data streams      
            .axis_dma_read_data         (axis_dma_read_data),
            
            //tcp send
            .m_axis_tx_metadata         (axis_tcp_tx_meta),
            .m_axis_tx_data             (axis_tcp_tx_data),
            .s_axis_tx_status           (axis_tcp_tx_status),
        
            //control reg
            .s_axis_put_data_to_net     (s_axis_put_data_to_net),
            .s_axis_get_data_cmd        (s_axis_get_data_cmd),
            .s_axis_put_data_cmd        (s_axis_put_data_cmd),


            .control_reg                (fpga_control_reg),
            .status_reg                 (fpga_status_reg)
        
            );

endmodule
