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

module tb_dma_read_data_to_fpga(

    );
    reg clk,rstn;

    // DMA Signals
        axis_mem_cmd    axis_dma_read_cmd();
        axi_stream      axis_dma_read_data();
        
    // memory cmd streams
        axis_mem_cmd    m_axis_write_cmd();
        // memory sts streams
        axis_mem_status     s_axis_write_sts();
        // memory data streams
        axi_stream   m_axis_write_data();
        
        axis_meta #(.WIDTH(96))     s_axis_put_data_cmd();
        
        
        reg[15:0][31:0]     fpga_control_reg;
        wire[1:0][31:0]     fpga_status_reg; 

    
        initial begin
            clk = 1'b1;
            rstn = 1'b0;
            fpga_control_reg[0] = 32'h1234_0000;
            fpga_control_reg[1] = 32'h0001_5678;
            fpga_control_reg[2] = 32'h1000_0000;
            fpga_control_reg[3] = 32'h0;
            #1000
            rstn = 1'b1;

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
    
        assign axis_dma_read_cmd.ready = 1'b1;
    
        assign axis_dma_read_data.valid = 1'b1;
        assign axis_dma_read_data.data = 1234;
        assign axis_dma_read_data.keep = 64'hffff_ffff_ffff_ffff;
        assign axis_dma_read_data.last = 1'b0;
    
    
        assign axis_mem_write_cmd.ready = 1'b1;
        assign axis_mem_write_data.ready = 1'b1;

        assign axis_mem_write_sts.valid = 1'b1;
        assign axis_mem_write_sts.data = 0;

        assign s_axis_put_data_cmd.valid = 1;
        assign s_axis_put_data_cmd.data = {32'h5678,32'h1234,32'h4000};
    
   
    
        dma_read_data_to_tcp dma_read_data_inst( 
    
            //user clock input
            .clk                        (clk),
            .rstn                       (rstn),
        
            //DMA Commands
            .axis_dma_read_cmd          (axis_dma_read_cmd),
            //DMA Data streams      
            .axis_dma_read_data         (axis_dma_read_data),
            
            //tcp send
            .m_axis_mem_write_cmd         (axis_mem_write_cmd),
            .s_axis_mem_write_sts             (axis_mem_write_sts),
            .m_axis_mem_write_data           (axis_mem_write_data),
        
            //control reg
            .s_axis_put_data_cmd           (s_axis_put_data_cmd),


            .control_reg                (fpga_control_reg),
            .status_reg                 (fpga_status_reg)
        
            );

endmodule
