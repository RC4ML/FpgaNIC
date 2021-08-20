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

module tb_hll(

    );
    reg clk,rstn,start;


    reg[15:0][31:0]      control_reg;
    reg[1:0][31:0]      status_reg;  
    reg[31:0]                  tx_data_cnt;    
    reg                         tx_data_valid;

    axis_mem_cmd                axis_dma_write_cmd();
    axi_stream                  axis_dma_write_data();    

    axis_meta #(.WIDTH(88))     app_axis_tcp_rx_meta();
    axi_stream #(.WIDTH(512))   app_axis_tcp_rx_data();
    axis_meta #(.WIDTH(48))     app_axis_tcp_tx_meta();
    axi_stream #(.WIDTH(512))   app_axis_tcp_tx_data();
    axis_meta #(.WIDTH(64))     app_axis_tcp_tx_status();
    
        initial begin
            clk = 1'b1;
            rstn = 1'b0;
            start = 1'b0;
            control_reg[0] = 1234;
            control_reg[1] = 2222;
            control_reg[2] = 8192;
            control_reg[3] = 5;
            control_reg[4] = 0;

            #500
            rstn = 1'b1;
            #100
            start = 1'b1;
            #10
            start = 1'b0;
            control_reg[4] = 1;
            
//            #1300
//            start = 1'b1;
//            #10
//            start = 1'b0;
            
//            #1300
//            start = 1'b1;
//            #10
//            start = 1'b0;            
            
//            #1300
//            start = 1'b1;
//            #10
//            start = 1'b0;            

//            #1300
//            start = 1'b1;
//            #10
//            start = 1'b0;
            
        end


        always #5 clk = ~clk;
    
        assign axis_dma_write_cmd.ready = 1;
        assign axis_dma_write_data.ready = 1;

        assign app_axis_tcp_tx_meta.ready = 1;
        assign app_axis_tcp_tx_data.ready = 1;



        always@(posedge clk)begin
            if(~rstn)begin
                tx_data_valid            <= 0;
            end
            else if(app_axis_tcp_rx_data.last)begin
                tx_data_valid            <= 0;
            end
            else if(start)begin
                tx_data_valid            <= 1;
            end
            else begin
                tx_data_valid            <= tx_data_valid;
            end
        end 
        
        
        always@(posedge clk)begin
            if(~rstn)begin
                tx_data_cnt        <= 0;
            end
            else if(app_axis_tcp_rx_data.last)begin
                tx_data_cnt       <= 0;
            end
            else if(app_axis_tcp_rx_data.ready & app_axis_tcp_rx_data.valid)begin
                tx_data_cnt       <= tx_data_cnt +16;
            end
            else begin
                tx_data_cnt       <= tx_data_cnt;
            end
        end 



        assign app_axis_tcp_rx_data.valid = tx_data_valid;
        assign app_axis_tcp_rx_data.data = {tx_data_cnt+15,tx_data_cnt+14,tx_data_cnt+13,tx_data_cnt+12,tx_data_cnt+11,tx_data_cnt+10,tx_data_cnt+9,tx_data_cnt+8,tx_data_cnt+7,tx_data_cnt+6,tx_data_cnt+5,tx_data_cnt+4,tx_data_cnt+3,tx_data_cnt+2,tx_data_cnt+1,tx_data_cnt};
        assign app_axis_tcp_rx_data.keep = 64'hffff_ffff_ffff_ffff;
        assign app_axis_tcp_rx_data.last = (tx_data_cnt == 32'h1ff_fff0) & app_axis_tcp_rx_data.valid & app_axis_tcp_rx_data.ready;;





//    hyperloglog hyperloglog_inst(
//        .clk(clk),
//        .rstn(rstn),
       
        
//        /* DMA INTERFACE */
//        .m_axis_dma_write_cmd(axis_dma_write_cmd),
//        .m_axis_dma_write_data(axis_dma_write_data),

//        .s_axis_read_data(app_axis_tcp_rx_data),
//       //app interface streams
//        .m_axis_tx_metadata          (app_axis_tcp_tx_meta),
//        .m_axis_tx_data              (app_axis_tcp_tx_data),
//        .s_axis_tx_status            (app_axis_tcp_tx_status),    
    
//        .s_axis_rx_metadata          (app_axis_tcp_rx_meta), 




//       //control reg
//        .control_reg(control_reg),
//        .status_reg(status_reg)     
    
//    );


         hyperloglog_ip hyperloglog_ip (
             .s_axis_input_tuple_TVALID      (app_axis_tcp_rx_data.valid),  // input wire s_axis_input_tuple_TVALID
             .s_axis_input_tuple_TREADY      (app_axis_tcp_rx_data.ready),  // output wire s_axis_input_tuple_TREADY
             .s_axis_input_tuple_TDATA       (app_axis_tcp_rx_data.data),    // input wire [511 : 0] s_axis_input_tuple_TDATA
             .s_axis_input_tuple_TKEEP       (64'hffff_ffff_ffff_ffff),    // input wire [63 : 0] s_axis_input_tuple_TKEEP
             .s_axis_input_tuple_TLAST       (app_axis_tcp_rx_data.last),    // input wire [0 : 0] s_axis_input_tuple_TLAST
             .m_axis_write_cmd_V_TVALID      (),  // output wire m_axis_write_cmd_V_TVALID
             .m_axis_write_cmd_V_TREADY      (1),  // input wire m_axis_write_cmd_V_TREADY
             .m_axis_write_cmd_V_TDATA       (),    // output wire [95 : 0] m_axis_write_cmd_V_TDATA
             .m_axis_write_data_TVALID       (),    // output wire m_axis_write_data_TVALID
             .m_axis_write_data_TREADY       (1),    // input wire m_axis_write_data_TREADY
             .m_axis_write_data_TDATA        (),      // output wire [31 : 0] m_axis_write_data_TDATA
             .m_axis_write_data_TKEEP        (),      // output wire [3 : 0] m_axis_write_data_TKEEP
             .m_axis_write_data_TLAST        (),      // output wire [0 : 0] m_axis_write_data_TLAST
             .regBaseAddr_V                  (0),                          // input wire [63 : 0] regBaseAddr_V
             .ap_clk                         (clk),                                        // input wire ap_clk
             .ap_rst_n                       (rstn)                                    // input wire ap_rst_n
           );

endmodule
