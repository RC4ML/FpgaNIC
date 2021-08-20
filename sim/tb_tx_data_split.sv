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

module tb_tx_data_split(

    );
    reg clk,rstn;
        
        axis_meta #(.WIDTH(48))     s_axis_tx_metadata();
        axi_stream #(.WIDTH(512))    s_axis_tx_data();
        axis_meta #(.WIDTH(64))     m_axis_tx_status();

        axis_meta #(.WIDTH(32))     m_axis_tx_metadata();
        axi_stream #(.WIDTH(512))    m_axis_tx_data();
        axis_meta #(.WIDTH(64))     s_axis_tx_status();

        reg tx_valid;
    
        initial begin
            clk = 1'b1;
            rstn = 1'b0;
            tx_valid = 1'b0;
            #500
            rstn = 1'b1;
            #500
            tx_valid = 1'b1;
            #100
            tx_valid = 1'b0;

            // fpga_control_reg[5] = 32'h1000;
            // fpga_control_reg[7] = 32'h4000;
            // fpga_control_reg[8] = 32'h0;
            // fpga_control_reg[9] = 32'h5;
            // #1000
            // rstn = 1'b1;
            // #1000
            // fpga_control_reg[8] = 32'h1;
            // #100
            // fpga_control_reg[8] = 32'h0;
            // fpga_control_reg[4] = 32'h15;
            // fpga_control_reg[5] = 32'h5000;
            // fpga_control_reg[7] = 32'h8000;            
            // #100
            // fpga_control_reg[8] = 32'h1;
        end
    
        always #5 clk = ~clk;
    
        assign m_axis_tx_data.ready = 1'b1;
    
        assign s_axis_tx_data.valid = 1'b1;
        assign s_axis_tx_data.data = 1234;
        assign s_axis_tx_data.keep = 64'hffff_ffff_ffff_ffff;
        assign s_axis_tx_data.last = 1'b0;

        assign s_axis_tx_metadata.valid = tx_valid;
        assign s_axis_tx_metadata.data = {32'h40000,16'h0};
    
    
        assign m_axis_tx_status.ready = 1'b1;
        assign m_axis_tx_metadata.ready = 1'b1;

        assign s_axis_tx_status.valid = 1;
        assign s_axis_tx_status.data = 0;


          

            tx_data_split inst_tx_data_split(
                .clk                    (clk),
                .rstn                   (rstn),
            
                //input tx_data
                .s_axis_tx_metadata     (s_axis_tx_metadata),
                .s_axis_tx_data         (s_axis_tx_data),
                .m_axis_tx_status       (m_axis_tx_status),
                //output tx_data splited
                .m_axis_tx_metadata     (m_axis_tx_metadata),
                .m_axis_tx_data         (m_axis_tx_data),
                .s_axis_tx_status       (s_axis_tx_status)
            
            
                );

endmodule
