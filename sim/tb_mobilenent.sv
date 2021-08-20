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

module tb_mobilenent(

    );
    reg clk,rstn;
    axi_mm #(.DATA_WIDTH(32),.ADDR_WIDTH(32))         m_axi_img();
    axi_mm #(.DATA_WIDTH(32),.ADDR_WIDTH(32))         m_axi_input();
    axi_mm #(.DATA_WIDTH(32),.ADDR_WIDTH(32))         m_axi_output();
        

    initial begin
        clk = 1'b1;
        rstn = 1'b0;
        #500
        rstn = 1'b1;
    end

    always #5 clk = ~clk;

    assign m_axi_img.awready     = 1;     
    assign m_axi_img.wready      = 1;   
    assign m_axi_img.bid         = 0;
    assign m_axi_img.bresp       = 1;
    assign m_axi_img.bvalid      = 1;
    assign m_axi_img.arready     = 1;
    assign m_axi_img.rid         = 0;
    assign m_axi_img.rresp       = 4'hf;
    assign m_axi_img.rdata       = 1234;                  
    assign m_axi_img.rvalid      = 1;
    assign m_axi_img.rlast       = 0; 

    assign m_axi_input.awready     = 1;     
    assign m_axi_input.wready      = 1;   
    assign m_axi_input.bid         = 0;
    assign m_axi_input.bresp       = 1;
    assign m_axi_input.bvalid      = 1;
    assign m_axi_input.arready     = 1;
    assign m_axi_input.rid         = 0;
    assign m_axi_input.rresp       = 4'hf;
    assign m_axi_input.rdata       = 1234;                  
    assign m_axi_input.rvalid      = 1;
    assign m_axi_input.rlast       = 1; 

    assign m_axi_output.awready     = 1;     
    assign m_axi_output.wready      = 1;   
    assign m_axi_output.bid         = 0;
    assign m_axi_output.bresp       = 1;
    assign m_axi_output.bvalid      = 1;
    assign m_axi_output.arready     = 1;
    assign m_axi_output.rid         = 0;
    assign m_axi_output.rresp       = 4'hf;
    assign m_axi_output.rdata       = 1234;                  
    assign m_axi_output.rvalid      = 1;
    assign m_axi_output.rlast       = 0;     

    mobilenet mobilenet_inst (
        .clk(clk),
        .rstn(rstn),
       
        
        /* DMA INTERFACE */
        .m_axi_img(m_axi_img),
        .m_axi_input(m_axi_input),
    
        .m_axi_output(m_axi_output)
   
   
       //control reg
       //  input wire[15:0][31:0]      control_reg,
       //  output wire[1:0][31:0]      status_reg     
    
    );

endmodule
