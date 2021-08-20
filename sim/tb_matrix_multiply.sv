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

module tb_matrix_multiply(

    );
    reg clk,rstn;

    axi_mm #(.DATA_WIDTH(256))       matrix();
   

    reg [15:0][31:0]    control_reg;
    reg [47:0]          tcp_rx_meta_data;
    reg                 mem_valid;

    reg tcp_rx_meta_valid;
    
        initial begin
            clk = 1'b1;
            rstn = 1'b0;
            // control_reg[0] = 1'b0;
            // control_reg[1] = 256;
            // control_reg[2] = 256*64;
            // control_reg[3] = 64;
            // control_reg[4] = 64;
            // control_reg[5] = 256;
            // control_reg[6] = 32'd0;
            control_reg[0] = 32'h10000000;
            control_reg[1] = 4;
            control_reg[2] = 256;
            control_reg[3] = 256*64;
            control_reg[4] = 64;
            control_reg[5] = 64;
            control_reg[6] = 256*64;
            control_reg[7] = 32'd0;  
            matrix.awready = 1;
            matrix.wready = 1;
            matrix.bid = 0;
            matrix.bresp = 3;
            matrix.bvalid = 1;
            matrix.arready = 1;
            matrix.rid = 0;
            matrix.rresp = 1;
            matrix.rdata = 1122;
            matrix.rvalid = 1;
            matrix.rlast = 1;                     
            #500
            rstn = 1'b1;
            #500
            control_reg[7] = 32'd1;


        end
    
        always #5 clk = ~clk;
    



//        assign matrix.awready = 1;
//        assign matrix.wready = 1;
//        assign matrix.bid = 0;
//        assign matrix.bresp = 01;
//        assign matrix.bvalid = 1;
//        assign matrix.arready = 1;
//        assign matrix.rid = 0;
//        assign matrix.rresp = 1;
//        assign matrix.rdata = 1122;
//        assign matrix.rvalid = 1;
//        assign matrix.rlast = 1;

        matrix_multiply1 matrix_multiply_inst( 
        //user clock input
        .clk(clk),
        .rstn(rstn),
        
        //dma memory streams
        .matrix(matrix),
    
        //control reg
        .control_reg(control_reg),
        .status_reg()
    
        
        );


endmodule
