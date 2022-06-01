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

module mpi_reduce_tx#(
    parameter   MAX_CLIENT      = 8
)( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,
	
    //ddr memory streams
    axi_mm.master    		    m_axis_mem_read,
  
    //tcp app interface streams
    axis_meta.master            m_axis_tcp_tx_meta,
    axi_stream.master           m_axis_tcp_tx_data,
    axis_meta.slave             s_axis_tcp_tx_status, 

    //control reg
    input wire[15:0]            session_id,
    input wire[31:0]            client_id,
    input wire[31:0]            client_num,
    input wire[31:0]            client_length,
    input wire[7:0]             token_mul,
    input wire[7:0]             token_divide_num,
    input wire                  start,
    output reg                  send_done,
    input wire[15:0]            recv_session_id,

    input wire                  send_start_empty,
    output reg                  send_start_rd_en,

    input wire                  send_req_empty,
    output reg                  send_req_rd_en,

    input wire                  send_enable_empty,
    output reg                  send_enable_rd_en    
	
	);


    parameter[31:0] PKG_LENGTH          = 32'h2000_0000;
    reg                                 send_cnt;
    reg [7:0]                           total_send_num,end_num;
    reg [2:0]                           start_cnt;
    reg [33:0]                          addr;
    reg [33:0]                          session_addr[1:0];
    reg                                 start_r;
    reg [31:0]                          data_cnt;
    reg [31:0]                          data_cnt_minus;


    parameter[3:0]      IDLE            = 4'h0,
                        SEND_START_CMD  = 4'h1,
                        SEND_START_DATA = 4'h2,
                        SEND_NEXT_START = 4'h3,
                        WAIT_SEND       = 4'h4,
                        SEND_WRITE_CMD  = 4'h5,
                        SEND_READ_ADDR  = 4'h6,
                        READ_DATA       = 4'h7,
                        END             = 4'h8,
                        WAIT_SEND_REQ   = 4'h9,
                        SEND_REQ_CMD    = 4'ha,
                        SEND_REQ_DATA   = 4'hb;


    reg [3:0]                           state;



    genvar i;
    generate
        for(i = 0; i < 2; i = i + 1) begin
            always@(posedge clk)begin
                session_addr[i]         <= PKG_LENGTH * i;
            end
        end
    endgenerate 


    assign m_axis_mem_read.awid        = 0;
    assign m_axis_mem_read.awlen       = 1; 
    assign m_axis_mem_read.awsize      = 3'b110;
    assign m_axis_mem_read.awburst     = 2'b01;
    assign m_axis_mem_read.awcache     = 0;
    assign m_axis_mem_read.awprot      = 0;
    assign m_axis_mem_read.awaddr      = 0;
    assign m_axis_mem_read.awlock      = 0;
    assign m_axis_mem_read.awvalid     = 0;
    assign m_axis_mem_read.awqos       = 0;
    assign m_axis_mem_read.awregion    = 0;                              
    assign m_axis_mem_read.wdata       = 0;
    assign m_axis_mem_read.wstrb       = 0;
    assign m_axis_mem_read.wlast       = 0;
    assign m_axis_mem_read.wvalid      = 0;
    assign m_axis_mem_read.bready      = 0;
    assign m_axis_mem_read.arid        = 0;
    assign m_axis_mem_read.araddr      = addr + session_addr[send_cnt];
    assign m_axis_mem_read.arlen       = 4'h7;
    assign m_axis_mem_read.arsize      = 3'b110;
    assign m_axis_mem_read.arburst     = 2'b01;
    assign m_axis_mem_read.arcache     = 0;
    assign m_axis_mem_read.arprot      = 0;
    assign m_axis_mem_read.arlock      = 0;
    assign m_axis_mem_read.arvalid     = (state == SEND_READ_ADDR);
    assign m_axis_mem_read.arqos       = 0;
    assign m_axis_mem_read.arregion    = 0;   



//////////////////////////////////////////////////////
    axi_stream                  axis_tcp_tx_data();
    reg[31:0]                   mem_read_count;
    wire                        m_axis_mem_read_last; 
    wire                        m_axis_mem_read_valid;
    wire                        m_axis_mem_read_ready;

    assign m_axis_mem_read.rready       = ((big_cnt <<< token_mul) > data_cnt) ? m_axis_mem_read_ready : 0;
    assign m_axis_mem_read_valid        = ((big_cnt <<< token_mul) > data_cnt) ? m_axis_mem_read.rvalid : 0;
    assign m_axis_mem_read_last         = (data_cnt == data_cnt_minus) & m_axis_mem_read.rvalid & m_axis_mem_read.rready;

    axis_data_fifo_512_d4096 mem_read_data_fifo (
        .s_axis_aresetn(rstn),          // input wire s_axis_aresetn
        .s_axis_aclk(clk),                // input wire s_axis_aclk
        .s_axis_tvalid(m_axis_mem_read_valid),            // input wire s_axis_tvalid
        .s_axis_tready(m_axis_mem_read_ready),            // output wire s_axis_tready
        .s_axis_tdata(m_axis_mem_read.rdata),              // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(64'hffff_ffff_ffff_ffff),              // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(m_axis_mem_read_last),              // input wire s_axis_tlast
        .m_axis_tvalid(axis_tcp_tx_data.valid),            // output wire m_axis_tvalid
        .m_axis_tready(axis_tcp_tx_data.ready),            // input wire m_axis_tready
        .m_axis_tdata(axis_tcp_tx_data.data),              // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_tcp_tx_data.keep),              // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_tcp_tx_data.last),              // output wire m_axis_tlast
        .axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
        .axis_rd_data_count(mem_read_count)  // output wire [31 : 0] axis_rd_data_count
    );


    assign m_axis_tcp_tx_data.valid             = ((state == SEND_START_DATA) || (state == SEND_REQ_DATA)) ? 1 : axis_tcp_tx_data.valid;
    assign axis_tcp_tx_data.ready               = ((state == SEND_START_DATA) || (state == SEND_REQ_DATA)) ? 0 : m_axis_tcp_tx_data.ready;
    assign m_axis_tcp_tx_data.data              = (state == SEND_START_DATA) ? 1 : ((state == SEND_REQ_DATA) ? 512'hb5b5a6a6 : axis_tcp_tx_data.data);
    assign m_axis_tcp_tx_data.keep              = 64'hffff_ffff_ffff_ffff;
    assign m_axis_tcp_tx_data.last              = ((state == SEND_START_DATA) || (state == SEND_REQ_DATA)) ? 1 : axis_tcp_tx_data.last;
    


    assign m_axis_tcp_tx_meta.data              = (state == SEND_START_CMD) ? {32'h40,session_id} : ((state == SEND_REQ_CMD) ? {32'h40,recv_session_id} : {client_length,session_id});
    assign m_axis_tcp_tx_meta.valid             = (state == SEND_WRITE_CMD) || (state == SEND_START_CMD) || (state == SEND_REQ_CMD) ;

    assign s_axis_tcp_tx_status.ready           = 1'b1;


    always@(posedge clk)begin
        if(~rstn)begin
            send_cnt                            <= 1'b0;
        end
        else if(axis_tcp_tx_data.last & axis_tcp_tx_data.valid & axis_tcp_tx_data.ready)begin  
            send_cnt                            <= ~send_cnt;        
        end
        else begin
            send_cnt                            <= send_cnt;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            total_send_num                      <= 1'b0;
        end
        else if(axis_tcp_tx_data.last & axis_tcp_tx_data.valid & axis_tcp_tx_data.ready)begin
            total_send_num                      <= total_send_num + 1'b1;
        end
        else begin
            total_send_num                      <= total_send_num;
        end
    end    

    always@(posedge clk)begin
        if(~rstn)begin
            addr                                <= 1'b0;            
        end
        else if(addr == client_length)begin
            addr                                <= 1'b0;
        end
        else if(m_axis_mem_read.arvalid & m_axis_mem_read.arready)begin
            addr                                <= addr + 512;            
        end        
        else begin
            addr                                <= addr;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            data_cnt                            <= 1'b0;            
        end
        else if((data_cnt == data_cnt_minus) & m_axis_mem_read.rvalid & m_axis_mem_read.rready)begin
            data_cnt                            <= 1'b0;
        end
        else if(m_axis_mem_read.rvalid & m_axis_mem_read.rready)begin
            data_cnt                            <= data_cnt + 1;            
        end        
        else begin
            data_cnt                            <= data_cnt;
        end
    end    




    always@(posedge clk)begin
        end_num                                 <= (2 * client_num);
        send_done                               <= state == END;
        start_r                                 <= start;
        data_cnt_minus                          <= (client_length>>>6)-1;
    end

/////////////////////////////token/////////////////////
    reg[31:0]                                   small_cnt;
    reg[31:0]                                   big_cnt;
    reg                                         token_en;

    always@(posedge clk)begin
        if(~rstn)begin
            small_cnt                           <= 1'b0;
        end
        else if(small_cnt == token_divide_num)begin
            small_cnt                           <= 1'b0;
        end
        else if(token_en)begin
            small_cnt                           <= small_cnt + 1'b1;
        end
        else begin
            small_cnt                           <= small_cnt;
        end
    end


    always@(posedge clk)begin
        if(~rstn)begin
            big_cnt                             <= 1'b0;
        end
        else if(~token_en)begin
            big_cnt                             <= 1'b0;
        end
        else if(small_cnt == token_divide_num)begin
            big_cnt                             <= big_cnt + 1'b1;
        end
        else begin
            big_cnt                             <= big_cnt;
        end
    end



/////////////////////////////////////////////////////

    always@(posedge clk)begin
        if(~rstn)begin
            state                               <= IDLE;
            send_enable_rd_en                   <= 1'b0;
            send_start_rd_en                    <= 1'b0;
            send_req_rd_en                      <= 1'b0;
            start_cnt                           <= 1'b0;
            token_en                            <= 1'b0;
        end
        else begin
            send_enable_rd_en                   <= 1'b0;
            send_start_rd_en                    <= 1'b0;
            send_req_rd_en                      <= 1'b0;
            case(state)
                IDLE:begin
                    if(start_r)begin
                        state                   <= SEND_START_CMD;
                    end
                    else begin
                        state                   <= IDLE;
                    end
                end
                SEND_START_CMD:begin
                    if(m_axis_tcp_tx_meta.valid & m_axis_tcp_tx_meta.ready)begin
                        start_cnt               <= start_cnt + 1'b1;
                        state                   <= SEND_START_DATA;
                    end
                    else begin
                        state                   <= SEND_START_CMD;
                    end                    
                end
                SEND_START_DATA:begin
                    if(m_axis_tcp_tx_data.valid & m_axis_tcp_tx_data.ready)begin
                        state                   <= SEND_NEXT_START;
                    end
                    else begin
                        state                   <= SEND_START_DATA;
                    end                                                  
                end
                SEND_NEXT_START:begin
                    if(~send_start_empty)begin
                        send_start_rd_en        <= 1'b1;
                        if(start_cnt == client_num)begin
                            state               <= WAIT_SEND_REQ;
                        end
                        else begin
                            state               <= SEND_START_CMD;
                        end
                    end
                    else begin
                        state                   <= SEND_NEXT_START;
                    end                    
                end
                WAIT_SEND_REQ:begin
                    if(~send_enable_empty)begin
                        state                   <= SEND_REQ_CMD;
                        send_enable_rd_en       <= 1'b1;
                    end
                    else begin
                        state                   <= WAIT_SEND_REQ;
                    end
                end    
                SEND_REQ_CMD:begin
                    if(m_axis_tcp_tx_meta.valid & m_axis_tcp_tx_meta.ready)begin
                        state                   <= SEND_REQ_DATA;
                    end
                    else begin
                        state                   <= SEND_REQ_CMD;
                    end                    
                end
                SEND_REQ_DATA:begin
                    if(m_axis_tcp_tx_data.valid & m_axis_tcp_tx_data.ready)begin
                        state                   <= WAIT_SEND;
                    end
                    else begin
                        state                   <= SEND_REQ_DATA;
                    end                    
                end
                WAIT_SEND:begin
                    if(~send_req_empty)begin
                        token_en                <= 1'b1;
                        state                   <= SEND_WRITE_CMD;
                        send_req_rd_en          <= 1'b1;
                    end
                    else begin
                        state                   <= WAIT_SEND;
                    end
                end                             
                SEND_WRITE_CMD:begin
                    if(m_axis_tcp_tx_meta.valid & m_axis_tcp_tx_meta.ready)begin
                        state                   <= SEND_READ_ADDR;
                    end
                    else begin
                        state                   <= SEND_WRITE_CMD;
                    end
                end                
                SEND_READ_ADDR:begin
                    if(addr == (client_length - 512) && m_axis_mem_read.arvalid & m_axis_mem_read.arready)begin
                        state                   <= READ_DATA;
                    end
                    else begin
                        state                   <= SEND_READ_ADDR;
                    end
                end
                READ_DATA:begin
                    if(axis_tcp_tx_data.last & axis_tcp_tx_data.valid & axis_tcp_tx_data.ready)begin
                        token_en                <= 1'b0;
                        if(total_send_num == end_num)begin
                            state               <= END;
                        end
                        else begin
                            state               <= WAIT_SEND_REQ;
                        end                        
                    end
                    else begin
                        state                   <= READ_DATA;
                    end
                end
                END:begin
                    state                       <= IDLE;
                end
            endcase    
        end
    end

 
    // ila_mpi_tx ila_mpi_tx (
    //     .clk(clk), // input wire clk
    
    
    //     .probe0(send_enable_empty), // input wire [0:0]  probe0  
    //     .probe1(send_enable_rd_en), // input wire [0:0]  probe1 
    //     .probe2(send_done), // input wire [0:0]  probe2 
    //     .probe3(send_cnt), // input wire [3:0]  probe3 
    //     .probe4(total_send_num), // input wire [3:0]  probe4 
    //     .probe5(state), // input wire [3:0]  probe5 
    //     .probe6(remain_len), // input wire [31:0]  probe6 
    //     .probe7(s_axis_mem_read_data.valid), // input wire [0:0]  probe7 
    //     .probe8(s_axis_mem_read_data.ready), // input wire [0:0]  probe8 
    //     .probe9(s_axis_mem_read_data.last) // input wire [0:0]  probe9
    // );

    ila_mpi_tx ila_mpi_tx (
        .clk(clk), // input wire clk
    
    
        .probe0(m_axis_mem_read.arvalid), // input wire [0:0]  probe0  
        .probe1(m_axis_mem_read.arready), // input wire [0:0]  probe1 
        .probe2(m_axis_mem_read.araddr), // input wire [31:0]  probe2 
        .probe3(m_axis_mem_read.rvalid), // input wire [0:0]  probe3 
        .probe4(m_axis_mem_read.rready), // input wire [0:0]  probe4 
        .probe5(m_axis_mem_read.rlast), // input wire [0:0]  probe5 
        .probe6(m_axis_mem_read.rdata[31:0]), // input wire [31:0]  probe6 
        .probe7(send_req_empty), // input wire [0:0]  probe7 
        .probe8(send_req_rd_en), // input wire [0:0]  probe8 
        .probe9(send_enable_empty), // input wire [0:0]  probe9 
        .probe10(send_enable_rd_en), // input wire [0:0]  probe10 
        .probe11({total_send_num,send_cnt}), // input wire [7:0]  probe11 
        .probe12(state), // input wire [3:0]  probe12
        .probe13(big_cnt)
    );


endmodule
