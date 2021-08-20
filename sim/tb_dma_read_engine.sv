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

module tb_dma_read_engine( 

    );
    reg                             clk;
    reg                             rstn;
    
// DMA Signals
    axis_mem_cmd    axis_dma_read_cmd();
    axi_stream      axis_dma_read_data();

    reg[15:0][31:0]                 control_reg;



    initial begin
        clk = 1;
        rstn = 0;
        control_reg[0] = 32'h1234_0000;
        control_reg[1] = 32'h1234;
        control_reg[4] = 32'h40000;
        control_reg[5] = 32'h10;
        control_reg[6] = 32'h8000;
        control_reg[7] = 32'h0;
        #1000
        rstn = 1;
        #200
        control_reg[7] = 32'h2;
    end

    always #5 clk = ~clk;

    assign axis_dma_read_cmd.ready = 1'b1;

    assign axis_dma_read_data.valid = 1'b1;
    assign axis_dma_read_data.keep = 64'hffff_ffff_ffff_ffff;
    assign axis_dma_read_data.data = 4567;
    assign axis_dma_read_data.last = 0;


    dma_read_engine dma_read_engine_inst( 
        .clk                        (clk),
        .rstn                       (rstn),
        
        //DMA Commands
        .m_axis_dma_read_cmd       (axis_dma_read_cmd),
    
        //DMA Data streams      
        .s_axis_dma_read_data      (axis_dma_read_data),
        
        .control_reg                (control_reg),
        .status_reg                 ()
    
        );

endmodule
