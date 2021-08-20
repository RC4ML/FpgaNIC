`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/21/2020 02:05:24 PM
// Design Name: 
// Module Name: tb_dma_write_data
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


module tb_dma_write_data_from_fpga(

    );
    reg clk,rstn;

// DMA Signals
    axis_mem_cmd    axis_dma_write_cmd();
    axi_stream      axis_dma_write_data();
    
    // memory cmd streams
    axis_mem_cmd    axis_mem_read_cmd();
    // memory sts streams
    axis_mem_status     axis_mem_read_sts();
    // memory data streams
    axi_stream    axis_mem_read_data();

    axis_meta #(.WIDTH(96))     s_axis_get_data_cmd();
    
    assign s_axis_get_data_cmd.valid = 1;
    assign s_axis_get_data_cmd.data = {32'h5678,32'h1234,32'h4000};
    
    reg[15:0][31:0]     fpga_control_reg;
    wire[1:0][31:0]     fpga_status_reg; 
    

    reg[15:0][31:0]     fpga_control_reg;
    wire[1:0][31:0]     fpga_status_reg; 

    reg                 notification_valid;
    reg [87:0]          notification_data;
    reg [511:0]         rx_data_data;
    reg                 rx_data_valid;
    reg                 rx_data_last;
    reg [15:0]          rx_length;


    initial begin
        clk = 1'b1;
        rstn = 1'b0;
        fpga_control_reg[0] = 32'h1234_0000;
        fpga_control_reg[1] = 32'h0001_5678;
        fpga_control_reg[2] = 32'h800_0000;
        fpga_control_reg[3] = 32'h0;
        #1000
        rstn = 1'b1;
    end

    always #5 clk = ~clk;

    // reg [31:0]  cnt;

    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         rx_data_valid <= 1'b0;
    //     end
    //     else if(axis_tcp_read_pkg.ready & axis_tcp_read_pkg.valid)begin
    //         rx_data_valid <= 1'b1;
    //     end        
    //     else if(axis_tcp_rx_data.last)begin
    //         rx_data_valid <= 1'b0;
    //     end
    //     else begin
    //         rx_data_valid <= rx_data_valid;
    //     end
    // end

    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         rx_data_data <= {480'b0,16'h400,16'b0};
    //     end
    //     else if(axis_tcp_rx_data.ready && axis_tcp_rx_data.valid)begin
    //         rx_data_data <= rx_data_data +1'b1;
    //     end
    //     else begin
    //         rx_data_data <= rx_data_data;
    //     end
    // end



    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         rx_length <= 1'b0;
    //     end
    //     else if(axis_tcp_read_pkg.ready & axis_tcp_read_pkg.valid)begin
    //         rx_length <= axis_tcp_read_pkg.data[31:16];
    //     end
    //     else begin
    //         rx_length <= rx_length;
    //     end
    // end


    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         cnt <= 1'b0;
    //     end
    //     else if(axis_tcp_rx_data.last)begin
    //         cnt <= 0;
    //     end
    //     else if(axis_tcp_rx_data.ready && axis_tcp_rx_data.valid)begin
    //         cnt <= cnt +1'b1;
    //     end
    //     else begin
    //         cnt <= cnt;
    //     end
    // end

    assign axis_dma_write_cmd.ready = 1'b1;
    assign axis_dma_write_data.ready = 1'b1;

    assign axis_mem_read_cmd.ready = 1;
    assign axis_mem_read_data.valid = 1;
    assign axis_mem_read_data.data = 1234;
    assign axis_mem_read_data.keep = 64'hffff_ffff_ffff_ffff;
    assign axis_mem_read_data.last = 0;//(cnt == ((rx_length >> 6) - 1)) && axis_tcp_rx_data.ready && axis_tcp_rx_data.valid;


    assign axis_mem_read_sts.valid = 1'b1;
    assign axis_mem_read_sts.data = 0;


    dma_write_data_from_tcp dma_write_data_inst( 
    
        //user clock input
        .clk                        (clk),
        .rstn                       (rstn),
    
        //DMA Commands
        .axis_dma_write_cmd         (axis_dma_write_cmd),
    
        //DMA Data streams      
        .axis_dma_write_data        (axis_dma_write_data),
    
        //tcp send
        .m_axis_mem_read_cmd       (axis_mem_read_cmd),
        .s_axis_mem_read_sts        (axis_mem_read_sts),
        
        .s_axis_mem_read_data       (axis_mem_read_data),
    
        //control reg
        .s_axis_get_data_cmd        (s_axis_get_data_cmd),
        .control_reg                (fpga_control_reg),
        .status_reg                 (fpga_status_reg)
    
        );


endmodule
