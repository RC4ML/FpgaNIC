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

module tb_dma_write_engine( 

    );
    reg                             clk;
    reg                             rstn;
    
// DMA Signals
    axis_mem_cmd    axis_dma_write_cmd();
    axi_stream      axis_dma_write_data();

    reg[15:0][31:0]                 control_reg;



    initial begin
        clk = 1;
        rstn = 0;
        control_reg[2] = 32'h1234_0000;
        control_reg[3] = 32'h1234;
        control_reg[4] = 32'h40000;
        control_reg[5] = 32'h10;
        control_reg[6] = 32'h8000;
        control_reg[7] = 32'h0;
        #1000
        rstn = 1;
        #200
        control_reg[7] = 32'h1;
    end

    always #5 clk = ~clk;

    assign axis_dma_write_cmd.ready = 1'b1;
    assign axis_dma_write_data.ready = 1'b1;


    dma_write_engine dma_write_engine_inst( 
        .clk                        (clk),
        .rstn                       (rstn),
        
        //DMA Commands
        .m_axis_dma_write_cmd       (axis_dma_write_cmd),
    
        //DMA Data streams      
        .m_axis_dma_write_data      (axis_dma_write_data),
        
        .control_reg                (control_reg),
        .status_reg                 ()
    
        );


endmodule
