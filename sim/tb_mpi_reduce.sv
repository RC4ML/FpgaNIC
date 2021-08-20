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

module tb_mpi_reduce(

    );
    reg clk,rstn;
        
    // memory cmd streams
    axis_mem_cmd        axis_mem_read_cmd();
    axis_mem_cmd        axis_mem_write_cmd();
    // memory sts streams
    axis_mem_status     axis_mem_read_sts();
    axis_mem_status     axis_mem_write_sts();
    // memory data streams
    axi_stream          axis_mem_read_data();
    axi_stream          axis_mem_write_data(); 

    // memory cmd streams
    axis_mem_cmd        axis_mem_dma_read_cmd();
    axis_mem_cmd        axis_mem_dma_write_cmd();
    // memory sts streams
    axis_mem_status     axis_mem_dma_read_sts();
    axis_mem_status     axis_mem_dma_write_sts();
    // memory data streams
    axi_stream          axis_mem_dma_read_data();
    axi_stream          axis_mem_dma_write_data(); 

    // memory cmd streams
    axis_mem_cmd        axis_dma_read_cmd();
    axi_stream          axis_dma_read_data();
    // memory cmd streams
    axis_mem_cmd        axis_dma_ctrl_read_cmd();
    axi_stream          axis_dma_ctrl_read_data(); 

    axis_meta #(.WIDTH(88))     app_axis_tcp_rx_meta();
    axi_stream #(.WIDTH(512))   app_axis_tcp_rx_data();
    axis_meta #(.WIDTH(48))     app_axis_tcp_tx_meta();
    axi_stream #(.WIDTH(512))   app_axis_tcp_tx_data();
    axis_meta #(.WIDTH(64))     app_axis_tcp_tx_status();

    axi_mm #(.DATA_WIDTH(256))       axis_mem_read0();
    axi_mm #(.DATA_WIDTH(256))       axis_mem_read1();
    axi_mm #(.DATA_WIDTH(256))       axis_mem_write0();
    axi_mm #(.DATA_WIDTH(256))       axis_mem_write1();    

    reg [15:0][31:0]    control_reg;
    reg [47:0]          tcp_rx_meta_data;
    reg                 mem_valid;

    reg tcp_rx_meta_valid;
    
        initial begin
            clk = 1'b1;
            rstn = 1'b0;
            control_reg[0] = 1'b0;
            control_reg[1] = 1'b0;
            control_reg[2] = 32'h3;
            control_reg[3] = 32'd4096;
            control_reg[4] = 32'd0;
            control_reg[5] = 32'd0;
            control_reg[6] = 32'd0;
            control_reg[7] = 32'd0;
            control_reg[8] = 32'd0;
            control_reg[9] = 32'h13;
            control_reg[10]= 32'h3;
            tcp_rx_meta_valid = 1'b0;
            mem_valid = 1'b0;
            #500
            rstn = 1'b1;
            #500
            control_reg[4] = 32'd1;
            #500
            tcp_rx_meta_valid = 1'b1;
            tcp_rx_meta_data = {32'd40,16'h0};
            #10
            tcp_rx_meta_data = {32'd4096,16'h0};
            #500
            mem_valid = 1;


        end
    
        always #5 clk = ~clk;
    
        reg             read_valid0;
        reg[31:0]       reed_cnt0;
        reg[31:0]       read_len0;

        always@(posedge clk)begin
            if(~rstn)begin
                read_valid0        <= 0;
            end
            else if(axis_mem_read_data.last)begin
                read_valid0       <= 0;
            end            
            else if(axis_mem_read_cmd.ready & axis_mem_read_cmd.valid)begin
                read_valid0       <= 1;
            end
            else begin
                read_valid0       <= read_valid0;
            end
        end

        always@(posedge clk)begin
            if(~rstn)begin
                read_len0        <= 0;
            end
            else if(axis_mem_read_cmd.ready & axis_mem_read_cmd.valid)begin
                read_len0       <= (axis_mem_read_cmd.length >>> 6) -1;
            end
            else begin
                read_len0       <= read_len0;
            end
        end

        always@(posedge clk)begin
            if(~rstn)begin
                reed_cnt0        <= 0;
            end
            else if(axis_mem_read_data.last)begin
                reed_cnt0        <= 0;
            end            
            else if(axis_mem_read_data.ready & axis_mem_read_data.valid)begin
                reed_cnt0       <= reed_cnt0 +1;
            end
            else begin
                reed_cnt0       <= reed_cnt0;
            end
        end        
//////////////////////////////////////////////////////////////
        reg             read_valid1;
        reg[31:0]       reed_cnt1;
        reg[31:0]       read_len1;

        always@(posedge clk)begin
            if(~rstn)begin
                read_valid1        <= 0;
            end
            else if(axis_dma_read_data.last)begin
                read_valid1       <= 0;
            end            
            else if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
                read_valid1       <= 1;
            end
            else begin
                read_valid1       <= read_valid1;
            end
        end

        always@(posedge clk)begin
            if(~rstn)begin
                read_len1        <= 0;
            end
            else if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
                read_len1       <= (axis_dma_read_cmd.length >>> 6) -1;
            end
            else begin
                read_len1       <= read_len1;
            end
        end

        always@(posedge clk)begin
            if(~rstn)begin
                reed_cnt1        <= 0;
            end
            else if(axis_dma_read_data.last)begin
                reed_cnt1        <= 0;
            end
            else if(axis_dma_read_data.ready & axis_dma_read_data.valid)begin
                reed_cnt1       <= reed_cnt1 +1;
            end
            else begin
                reed_cnt1       <= reed_cnt1;
            end
        end     
//////////////////////////////////////////////////////////////
        reg             read_valid2;
        reg[31:0]       reed_cnt2;
        reg[31:0]       read_len2;

        always@(posedge clk)begin
            if(~rstn)begin
                read_valid2        <= 0;
            end
            else if(app_axis_tcp_rx_data.last)begin
                read_valid2       <= 0;
            end            
            else if(app_axis_tcp_rx_meta.ready & app_axis_tcp_rx_meta.valid)begin
                read_valid2       <= 1;
            end
            else begin
                read_valid2       <= read_valid2;
            end
        end

        always@(posedge clk)begin
            if(~rstn)begin
                read_len2        <= 0;
            end
            else if(app_axis_tcp_rx_meta.ready & app_axis_tcp_rx_meta.valid)begin
                read_len2       <= (app_axis_tcp_rx_meta.data[47:16] >>> 6) -1;
            end
            else begin
                read_len2       <= read_len2;
            end
        end

        always@(posedge clk)begin
            if(~rstn)begin
                reed_cnt2        <= 0;
            end
            else if(app_axis_tcp_rx_data.last)begin
                reed_cnt2       <= 0;
            end
            else if(app_axis_tcp_rx_data.ready & app_axis_tcp_rx_data.valid)begin
                reed_cnt2       <= reed_cnt2 +1;
            end
            else begin
                reed_cnt2       <= reed_cnt2;
            end
        end 


        assign axis_mem_write_cmd.ready = 1'b1;
        assign axis_mem_write_sts.valid = 1'b1;
        assign axis_mem_write_sts.data = 1'b1;
        assign axis_mem_write_data.ready = 1'b1;

        // assign axis_mem_write_cmd[1].ready = 1'b1;
        // assign axis_mem_write_sts[1].valid = 1'b1;
        // assign axis_mem_write_sts[1].data = 1'b1;
        // assign axis_mem_write_data[1].data = 1'b1;        

        assign axis_mem_read_cmd.ready = 1'b1;
        assign axis_mem_read_sts.valid = 1'b1;
        assign axis_mem_read_sts.data = 1'b1;

        assign axis_mem_read_data.valid = read_valid0;
        assign axis_mem_read_data.data = 32'h11aa;
        assign axis_mem_read_data.keep = 64'hffff_ffff_ffff_ffff;
        assign axis_mem_read_data.last = axis_mem_read_data.valid & axis_mem_read_data.ready && (reed_cnt0 == read_len0);

        // assign axis_mem_read_cmd[1].ready = 1'b1;
        // assign axis_mem_read_sts[1].valid = 1'b1;
        // assign axis_mem_read_sts[1].data = 1'b1;

        assign axis_dma_read_cmd.ready = 1'b1;
        assign axis_dma_read_data.valid = read_valid1;
        assign axis_dma_read_data.data = 32'h22aa;
        assign axis_dma_read_data.keep = 64'hffff_ffff_ffff_ffff;
        assign axis_dma_read_data.last = axis_dma_read_data.valid & axis_dma_read_data.ready && (reed_cnt1 == read_len1);

        



        assign app_axis_tcp_tx_status.valid = 1'b1;
        assign app_axis_tcp_tx_status.data = 1;
        assign app_axis_tcp_tx_meta.ready = 1'b1;
        assign app_axis_tcp_tx_data.ready = 1'b1;

        assign app_axis_tcp_rx_meta.valid = tcp_rx_meta_valid;
        assign app_axis_tcp_rx_meta.data = tcp_rx_meta_data;
    
        assign app_axis_tcp_rx_data.valid = read_valid2;
        assign app_axis_tcp_rx_data.data = 32'h33aa;
        assign app_axis_tcp_rx_data.keep = 64'hffff_ffff_ffff_ffff;
        assign app_axis_tcp_rx_data.last = app_axis_tcp_rx_data.valid & app_axis_tcp_rx_data.ready && (reed_cnt2 == read_len2);


        assign axis_mem_read0.awready = 1;
        assign axis_mem_read0.wready = 1;
        assign axis_mem_read0.bid = 0;
        assign axis_mem_read0.bresp = 01;
        assign axis_mem_read0.bvalid = 1;
        assign axis_mem_read0.arready = 1;
        assign axis_mem_read0.rid = 0;
        assign axis_mem_read0.rresp = 1;
        assign axis_mem_read0.rdata = 1122;
        assign axis_mem_read0.rvalid = mem_valid;
        assign axis_mem_read0.rlast = 1;
          
        assign axis_mem_read1.awready = 1;
        assign axis_mem_read1.wready = 1;
        assign axis_mem_read1.bid = 0;
        assign axis_mem_read1.bresp = 01;
        assign axis_mem_read1.bvalid = 1;
        assign axis_mem_read1.arready = 1;
        assign axis_mem_read1.rid = 0;
        assign axis_mem_read1.rresp = 1;
        assign axis_mem_read1.rdata = 2233;
        assign axis_mem_read1.rvalid = mem_valid;
        assign axis_mem_read1.rlast = 1;

        assign axis_mem_write0.awready = 1;
        assign axis_mem_write0.wready = 1;
        assign axis_mem_write0.bid = 0;
        assign axis_mem_write0.bresp = 01;
        assign axis_mem_write0.bvalid = 1;
        assign axis_mem_write0.arready = 1;
        assign axis_mem_write0.rid = 0;
        assign axis_mem_write0.rresp = 1;
        assign axis_mem_write0.rdata = 3344;
        assign axis_mem_write0.rvalid = mem_valid;
        assign axis_mem_write0.rlast = 1;

        assign axis_mem_write1.awready = 1;
        assign axis_mem_write1.wready = 1;
        assign axis_mem_write1.bid = 0;
        assign axis_mem_write1.bresp = 01;
        assign axis_mem_write1.bvalid = 1;
        assign axis_mem_write1.arready = 1;
        assign axis_mem_write1.rid = 0;
        assign axis_mem_write1.rresp = 1;
        assign axis_mem_write1.rdata = 4455;
        assign axis_mem_write1.rvalid = mem_valid;
        assign axis_mem_write1.rlast = 1;

    mpi_reduce_control mpi_reduce_control_inst( 
        //user clock input
        .clk(clk),
        .rstn(rstn),
        
        //dma memory streams
        .m_axis_dma_write_cmd(axis_mem_write_cmd),
        .m_axis_dma_write_data(axis_mem_write_data),
        .m_axis_dma_read_cmd(axis_mem_read_cmd),
        .s_axis_dma_read_data(axis_mem_read_data),    
    
        //dma memory streams
        .m_axis_mem_read0(axis_mem_read0),
        // .m_axis_mem_read1(axis_mem_read1),
        .m_axis_mem_write0(axis_mem_write0),
        // .m_axis_mem_write1(axis_mem_write1),     
    
        //tcp app interface streams
        .app_axis_tcp_tx_meta(app_axis_tcp_tx_meta),
        .app_axis_tcp_tx_data(app_axis_tcp_tx_data),
        .app_axis_tcp_tx_status(app_axis_tcp_tx_status),    
    
        .app_axis_tcp_rx_meta(app_axis_tcp_rx_meta),
        .app_axis_tcp_rx_data(app_axis_tcp_rx_data),
    
        //control reg
        .control_reg(control_reg),
        .status_reg()
    
        
        );
                
endmodule
