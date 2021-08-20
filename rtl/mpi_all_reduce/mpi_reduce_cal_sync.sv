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

module mpi_reduce_cal#(
    parameter   MAX_CLIENT      = 8
)( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,
	
    //ddr memory streams
    axi_mm.master    		    m_axis_mem_write,
    //DMA Commands
    axis_mem_cmd.master    		m_axis_dma_read_cmd,
    axi_stream.slave    		s_axis_dma_read_data, 
    //DMA Commands
    axis_mem_cmd.master    		m_axis_dma_write_cmd,
    axi_stream.master    		m_axis_dma_write_data,     
    
  
    //tcp Commands     
    axis_meta.slave             s_axis_tcp_rx_meta,
    axi_stream.slave            s_axis_tcp_rx_data,

    //control reg
    input wire[15:0]            session_id,
    input wire[31:0]            client_id,
    input wire[31:0]            client_num,
    input wire[31:0]            client_length,
    input wire                  recv_start,
    input wire[63:0]            dma_read_base_addr,
    input wire[63:0]            dma_write_base_addr,

    output reg[15:0]            recv_session_id,
    output wire                 send_enable_empty,
    input wire                  send_enable_rd_en,
    output wire                 send_start_empty,
    input wire                  send_start_rd_en,
    output wire                 send_req_empty,
    input wire                  send_req_rd_en,    
    output reg                  start_cal_flag,
    output reg                  mpi_reduce_done
	
	);

    parameter[31:0] PKG_LENGTH          = 32'h2000_0000;
    reg [7:0]                           recv_cnt,total_recv_num;
    reg [31:0]                          remain_len,length,addr;
    reg [31:0]                          session_addr[MAX_CLIENT-1:0];
    reg                                 data_ready;
    reg                                 mem_flag;
    reg                                 recv_end;
    reg                                 clear_wr_data_done;
    reg [7:0]                           cnt;
    reg [31:0]                          data_cnt_minus,data_cnt;
    reg                                 recv_start_r;
    reg                                 session_fifo_rd_pre,session_fifo_rd_pre_r;
    reg                                 mem_fifo_rd_pre,mem_fifo_rd_pre_r;


    parameter[3:0]      IDLE                    = 4'h0,
                        RECV_START_INFO         = 4'he,
                        RECV_NEXT_INFO          = 4'h1, 
                        FIRST_DMA_READ_CMD      = 4'h2,
                        FIRST_MEM_WRITE_ADDR    = 4'h3,
                        START_CAL               = 4'h4,
                        START_CAL_R             = 4'he,
                        SEND_READ_CMD           = 4'h5,
                        WAIT_RD_EN              = 4'h6,
                        CAL_START               = 4'h7,
                        CAL                     = 4'h8,
                        SEND_NEXT_CMD           = 4'h9,
                        SEND_RESULT_CMD         = 4'ha,
                        SEND_RESULT             = 4'hb,
                        SEND_NEXT_RESULT_CMD    = 4'hc,
                        END                     = 4'hd;
                      



    reg [3:0]                           state;
    reg [3:0]                           start_state;


    reg             [31:0]              wr_ops;
    reg             [7 :0]              burst_inc;
    reg                                 wr_data_done;  
    reg             [31:0]              num_mem_ops_minus_1;  
    reg [33:0]                          mem_addr;
    reg                                 mem_awvalid,mem_awvalid_en;
    reg                                 first_write_flag;
    reg                                 mpi_reduce_process;

    genvar i;
    generate
        for(i = 0; i < MAX_CLIENT; i = i + 1) begin
            always@(posedge clk)begin
                session_addr[i]         <=client_length*i;
            end
        end
    endgenerate 


    axi_stream                          axis_mem_write_data();

    assign m_axis_mem_write.awid        = 0;
    assign m_axis_mem_write.awlen       = 4'h7; 
    assign m_axis_mem_write.awsize      = 3'b110;
    assign m_axis_mem_write.awburst     = 2'b01;
    assign m_axis_mem_write.awcache     = 0;
    assign m_axis_mem_write.awprot      = 0;
    assign m_axis_mem_write.awaddr      = mem_flag ? (mem_addr + PKG_LENGTH) : mem_addr;
    assign m_axis_mem_write.awlock      = 0;
    assign m_axis_mem_write.awvalid     = mem_awvalid;
    assign m_axis_mem_write.awqos       = 0;
    assign m_axis_mem_write.awregion    = 0;                              
    assign m_axis_mem_write.wdata       = first_write_flag ? s_axis_dma_read_data.data : axis_mem_write_data.data;
    assign m_axis_mem_write.wstrb       = 64'hffff_ffff_ffff_ffff;
    assign m_axis_mem_write.wlast       = (burst_inc == 4'h7)&m_axis_mem_write.wvalid & m_axis_mem_write.wready;
    assign m_axis_mem_write.wvalid      = first_write_flag ? s_axis_dma_read_data.valid : axis_mem_write_data.valid;
    assign m_axis_mem_write.bready      = 1;
    assign m_axis_mem_write.arid        = 0;
    assign m_axis_mem_write.araddr      = 0;
    assign m_axis_mem_write.arlen       = 0;
    assign m_axis_mem_write.arsize      = 0;
    assign m_axis_mem_write.arburst     = 0;
    assign m_axis_mem_write.arcache     = 0;
    assign m_axis_mem_write.arprot      = 0;
    assign m_axis_mem_write.arlock      = 0;
    assign m_axis_mem_write.arvalid     = 0;
    assign m_axis_mem_write.arqos       = 0;
    assign m_axis_mem_write.arregion    = 0;   
    assign m_axis_mem_write.rready      = 0; 

    always @(posedge clk)begin
        if (~rstn)begin
            mem_addr                    <= 8'b0;
        end
        else if (m_axis_mem_write.awvalid & m_axis_mem_write.awready)begin
            mem_addr                    <= mem_addr + 512;
            if (mem_addr == (client_length - 512)) begin
                mem_addr                <= 8'b0;
            end
        end
        else begin
            mem_addr                    <= mem_addr;
        end
    end

    always @(posedge clk)begin
        if (~rstn)begin
            mem_awvalid                 <= 1'b0;
        end
        else if((mem_addr == (client_length - 512)) && m_axis_mem_write.awvalid & m_axis_mem_write.awready) begin
            mem_awvalid                 <= 1'b0;
        end 
        else if (mem_awvalid_en)begin
            mem_awvalid                 <= 1'b1;
        end
        else begin
            mem_awvalid                 <= mem_awvalid;
        end
    end

    always @(posedge clk)begin
        if (~rstn)begin
            burst_inc                   <= 8'b0;
            wr_ops                      <= 32'b0;
            wr_data_done                <= 1'b0;            
        end
        else if(clear_wr_data_done) begin
            burst_inc                   <= 8'b0;
            wr_data_done                <= 1'b0;
            wr_ops                      <= 32'b0;
        end        
        else if (m_axis_mem_write.wvalid & m_axis_mem_write.wready)begin
            burst_inc                   <= burst_inc + 8'b1;
            if (burst_inc == 4'h7) begin
                burst_inc               <= 8'b0;
                wr_ops                  <= wr_ops + 1'b1;
                if (wr_ops == num_mem_ops_minus_1)begin
                    wr_data_done        <= 1'b1;
                end                
            end
        end
    end

////////////////////////////////////////////tcp_rx

    wire                         axis_tcp_rx_data_valid;
    wire                         axis_tcp_rx_data_ready;
    axi_stream                  axis_session_data();
    wire[31:0]                   session_data_count;

    assign s_axis_tcp_rx_meta.ready           = 1'b1;

    always@(posedge clk)begin
        if(~rstn)begin
            recv_session_id                 <= 0;
        end
        else if(s_axis_tcp_rx_meta.ready & s_axis_tcp_rx_meta.valid && (s_axis_tcp_rx_meta.data[15:0] != session_id))begin
            recv_session_id                 <= s_axis_tcp_rx_meta.data[15:0];
        end
        else begin
            recv_session_id                 <= recv_session_id;
        end
    end

//////////////dma cmd buffer ///////////
	reg 									send_req_wr_en;

    always@(posedge clk)begin
        if(~rstn)begin
            send_req_wr_en                  <= 1'b0;
        end
        else if(s_axis_tcp_rx_meta.ready & s_axis_tcp_rx_meta.valid && (s_axis_tcp_rx_meta.data[15:0] == session_id))begin
            send_req_wr_en                  <= 1'b1;
        end 
        else begin
            send_req_wr_en                  <= 1'b0;
        end
    end

	blockram_fifo #( 
		.FIFO_WIDTH      ( 1 ), //64 
		.FIFO_DEPTH_BITS ( 8 )  //determine the size of 16  13
	) inst_send_req_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (send_req_wr_en     ), //or one cycle later...
	.din        (1 ),
	.almostfull (), //back pressure to  

	//reading side.....
	.re         (send_req_rd_en     ),
	.dout       (   ),
	.valid      (	),
	.empty      (send_req_empty     ),
	.count      (   )
    );
    


////////////////////////////////////////////////////////






    assign axis_tcp_rx_data_valid           = ((start_state == RECV_START_INFO) || (s_axis_tcp_rx_data.data[31:0] == 32'hb5b5a6a6)) ? 0 : (mpi_reduce_process ? s_axis_tcp_rx_data.valid : 0);
    assign s_axis_tcp_rx_data.ready         = (start_state == RECV_START_INFO) ? 1 : (mpi_reduce_process ? axis_tcp_rx_data_ready : 0);

    axis_data_fifo_512_d4096 session_data_fifo (
        .s_axis_aresetn(rstn),          // input wire s_axis_aresetn
        .s_axis_aclk(clk),                // input wire s_axis_aclk
        .s_axis_tvalid(axis_tcp_rx_data_valid),            // input wire s_axis_tvalid
        .s_axis_tready(axis_tcp_rx_data_ready),            // output wire s_axis_tready
        .s_axis_tdata(s_axis_tcp_rx_data.data),              // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_tcp_rx_data.keep),              // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_tcp_rx_data.last),              // input wire s_axis_tlast
        .m_axis_tvalid(axis_session_data.valid),            // output wire m_axis_tvalid
        .m_axis_tready(axis_session_data.ready),            // input wire m_axis_tready
        .m_axis_tdata(axis_session_data.data),              // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_session_data.keep),              // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_session_data.last),              // output wire m_axis_tlast
        .axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
        .axis_rd_data_count(session_data_count)  // output wire [31 : 0] axis_rd_data_count
    );

    assign axis_session_data.ready              = data_ready & axis_cal_data.ready;
//////////////////////////////////////////////////////
    axi_stream                  axis_mem_read_data();
    wire[31:0]                   dma_read_count;

    reg[31:0]                   dma_cmd_length;
    reg[31:0]                   dma_data_cnt;
    wire                        dma_read_data_last;
    wire                        dma_read_data_valid;
    wire                        dma_read_data_ready;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_cmd_length                  <= 0;
        end
        else if(m_axis_dma_read_cmd.ready & m_axis_dma_read_cmd.valid)begin
            dma_cmd_length                  <= (m_axis_dma_read_cmd.length >>> 6) -1;
        end
        else begin
            dma_cmd_length                  <= dma_cmd_length;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            dma_data_cnt                    <= 0;
        end
        else if(dma_read_data_last)begin
            dma_data_cnt                    <= 0;
        end            
        else if(s_axis_dma_read_data.ready & s_axis_dma_read_data.valid)begin
            dma_data_cnt                    <= dma_data_cnt +1;
        end
        else begin
            dma_data_cnt                    <= dma_data_cnt;
        end
    end  

    always@(posedge clk)begin
        if(~rstn)begin
            first_write_flag                    <= 0;
        end
        else if(state == FIRST_DMA_READ_CMD)begin
            first_write_flag                    <= 1;
        end            
        else if(dma_read_data_last)begin
            first_write_flag                    <= 0;
        end
        else begin
            first_write_flag                   <= first_write_flag;
        end
    end     



    assign dma_read_data_last = s_axis_dma_read_data.valid & s_axis_dma_read_data.ready && (dma_data_cnt == dma_cmd_length);

    assign dma_read_data_valid = first_write_flag ? 0 : s_axis_dma_read_data.valid;

    assign s_axis_dma_read_data.ready = first_write_flag ? m_axis_mem_write.wready : dma_read_data_ready;

    axis_data_fifo_512_d4096 dma_read_data_fifo (
        .s_axis_aresetn(rstn),          // input wire s_axis_aresetn
        .s_axis_aclk(clk),                // input wire s_axis_aclk
        .s_axis_tvalid(dma_read_data_valid),            // input wire s_axis_tvalid
        .s_axis_tready(dma_read_data_ready),            // output wire s_axis_tready
        .s_axis_tdata(s_axis_dma_read_data.data),              // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_dma_read_data.keep),              // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(dma_read_data_last),              // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_read_data.valid),            // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_read_data.ready),            // input wire m_axis_tready
        .m_axis_tdata(axis_mem_read_data.data),              // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_read_data.keep),              // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_read_data.last),              // output wire m_axis_tlast
        .axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
        .axis_rd_data_count(dma_read_count)  // output wire [31 : 0] axis_rd_data_count
    );

    assign axis_mem_read_data.ready             = data_ready & axis_cal_data.ready;

////////////////////////////////////////////////////////////////
    axi_stream                                  axis_cal_data();
    wire                                        mem_write_data_fifo_ready; 
    wire                                        mem_write_data_fifo_almost_full; 
    reg [511:0]                                 cal_data;
    reg                                         cal_data_valid;
    reg                                         cal_data_last;
    wire[31:0]                                  cal_read_count;

    always@(posedge clk)begin
        if(~rstn)begin
            cal_data                            <= 1'b0;
            cal_data_valid                      <= 1'b0;
            cal_data_last                       <= 1'b0;
        end
        else if(axis_mem_read_data.ready & axis_mem_read_data.valid)begin
            cal_data                            <= axis_mem_read_data.data + axis_session_data.data;
            cal_data_valid                      <= 1'b1;
            cal_data_last                       <= axis_mem_read_data.last;
        end
        else begin
            cal_data                            <= cal_data;
            cal_data_valid                      <= 1'b0;
            cal_data_last                       <= 1'b0;
        end
    end    

    always@(posedge clk)begin
        if(~rstn)begin
            data_cnt                            <= 1'b0;
        end
        else if(axis_cal_data.last)begin
            data_cnt                            <= 1'b0;
        end
        else if(axis_cal_data.ready & axis_cal_data.valid)begin
            data_cnt                            <= data_cnt + 1'b1;
        end
        else begin
            data_cnt                            <= data_cnt;
        end
    end

    axis_data_fifo_512_d4096 mem_write_data_fifo (
        .s_axis_aresetn(rstn),          // input wire s_axis_aresetn
        .s_axis_aclk(clk),                // input wire s_axis_aclk
        .s_axis_tvalid(axis_cal_data.valid),            // input wire s_axis_tvalid
        .s_axis_tready(mem_write_data_fifo_ready),            // output wire s_axis_tready
        .s_axis_tdata(axis_cal_data.data),              // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_cal_data.keep),              // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_cal_data.last),              // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_write_data.valid),            // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_write_data.ready),            // input wire m_axis_tready
        .m_axis_tdata(axis_mem_write_data.data),              // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_write_data.keep),              // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_write_data.last),              // output wire m_axis_tlast
        .axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
        .axis_rd_data_count(cal_read_count),  // output wire [31 : 0] axis_rd_data_count
        .prog_full(mem_write_data_fifo_almost_full)                    // output wire prog_full
    );

    assign axis_mem_write_data.ready            = m_axis_mem_write.wready;
/////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

    wire                                        dma_write_data_fifo_ready; 
    wire                                        dma_write_data_fifo_almost_full; 
    wire                                        dma_write_data_fifo_valid;
    wire[31:0]                                  dma_wr_read_count;



    assign axis_cal_data.valid                  = (state == SEND_RESULT) ? (axis_session_data.valid & data_ready & axis_cal_data.ready) : cal_data_valid;
    assign axis_cal_data.data                   = (state == SEND_RESULT) ? axis_session_data.data : cal_data;
    assign axis_cal_data.keep                   = 64'hffff_ffff_ffff_ffff;
    assign axis_cal_data.last                   = (state == SEND_RESULT) ? ((data_cnt == data_cnt_minus) && axis_cal_data.ready & axis_cal_data.valid) : cal_data_last;
    assign axis_cal_data.ready                  = mem_write_data_fifo_ready & dma_write_data_fifo_ready;

    assign dma_write_data_fifo_valid            = (total_recv_num > client_num) ? ((state == SEND_RESULT) ? (axis_session_data.valid & data_ready & axis_cal_data.ready) : cal_data_valid) : 0;

    axis_data_fifo_512_d4096 dma_write_data_fifo (
        .s_axis_aresetn(rstn),          // input wire s_axis_aresetn
        .s_axis_aclk(clk),                // input wire s_axis_aclk
        .s_axis_tvalid(dma_write_data_fifo_valid),            // input wire s_axis_tvalid
        .s_axis_tready(dma_write_data_fifo_ready),            // output wire s_axis_tready
        .s_axis_tdata(axis_cal_data.data),              // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_cal_data.keep),              // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_cal_data.last),              // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_dma_write_data.valid),            // output wire m_axis_tvalid
        .m_axis_tready(m_axis_dma_write_data.ready),            // input wire m_axis_tready
        .m_axis_tdata(m_axis_dma_write_data.data),              // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_dma_write_data.keep),              // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_dma_write_data.last),              // output wire m_axis_tlast
        .axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
        .axis_rd_data_count(dma_wr_read_count),  // output wire [31 : 0] axis_rd_data_count
        .prog_full(dma_write_data_fifo_almost_full)                    // output wire prog_full
    );












	//////////////dma cmd buffer ///////////
	reg 									send_enable_wr_en;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 1 ), //64 
		.FIFO_DEPTH_BITS ( 8 )  //determine the size of 16  13
	) inst_send_enable_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (send_enable_wr_en     ), //or one cycle later...
	.din        (1 ),
	.almostfull (), //back pressure to  

	//reading side.....
	.re         (send_enable_rd_en     ),
	.dout       (   ),
	.valid      (	),
	.empty      (send_enable_empty     ),
	.count      (   )
    );
    


////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
	//////////////dma cmd buffer ///////////
	reg 									send_start_wr_en;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 1 ), //64 
		.FIFO_DEPTH_BITS ( 8 )  //determine the size of 16  13
	) inst_send_start_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (send_start_wr_en     ), //or one cycle later...
	.din        (1 ),
	.almostfull (), //back pressure to  

	//reading side.....
	.re         (send_start_rd_en     ),
	.dout       (   ),
	.valid      (	),
	.empty      (send_start_empty     ),
	.count      (   )
    );
    


////////////////////////////////////////////////////////
    reg [7:0]                                   dma_total_cnt;    
    reg [3:0]                                   dma_cmd_state;   
    reg                                         dma_add;   
    reg                                         dma_read_valid;
    reg                                         dma_write_valid;     


    assign m_axis_dma_write_cmd.address         = dma_write_base_addr + addr;
    assign m_axis_dma_write_cmd.length          = client_length;
    assign m_axis_dma_write_cmd.valid           = dma_write_valid;

    assign m_axis_dma_read_cmd.address          = dma_read_base_addr + addr;
    assign m_axis_dma_read_cmd.length           = client_length;
    assign m_axis_dma_read_cmd.valid            = dma_read_valid;

    always@(posedge clk)begin
        if(~rstn)begin
            recv_cnt                            <= 1'b0;
        end
        else if(recv_start)begin
            recv_cnt                            <= client_id;
        end
        else if(dma_add)begin
            if(recv_cnt == 0 )begin
                recv_cnt                        <= client_num;
            end
            else begin
                recv_cnt                        <= recv_cnt - 1;
            end            
        end
        else begin
            recv_cnt                            <= recv_cnt;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            dma_total_cnt                       <= 1'b0;
        end
        else if(dma_add)begin
            dma_total_cnt                       <= dma_total_cnt + 1'b1;
        end
        else begin
            dma_total_cnt                       <= dma_total_cnt;
        end
    end 

    always@(posedge clk)begin
        if(~rstn)begin
            addr                                <= 1'b0;            
        end       
        else begin
            addr                                <= session_addr[recv_cnt];
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            dma_cmd_state                       <= IDLE;
            dma_add                             <= 0;
            dma_read_valid                      <= 0;
            dma_write_valid                     <= 0;
        end
        else begin
            dma_add                             <= 0;
            case(dma_cmd_state)                
                IDLE:begin
                    if(start_cal_flag)begin
                        dma_cmd_state           <= SEND_READ_CMD;
                    end
                    else begin
                        dma_cmd_state           <= IDLE;
                    end
                end
                SEND_READ_CMD:begin
                    dma_read_valid              <= 1;
                    if(m_axis_dma_read_cmd.valid & m_axis_dma_read_cmd.ready)begin
                        dma_read_valid          <= 0;
                        dma_add                 <= 1;
                        dma_cmd_state           <= SEND_NEXT_CMD;
                    end
                    else begin
                        dma_cmd_state           <= SEND_READ_CMD;
                    end
                end             
                SEND_NEXT_CMD:begin
                    if(dma_total_cnt == client_num)begin
                        dma_cmd_state           <= WAIT_RD_EN;
                    end 
                    else begin
                        dma_cmd_state           <= SEND_READ_CMD;
                    end            
                end      
                WAIT_RD_EN:begin
                    if(state == SEND_RESULT_CMD)begin
                        dma_cmd_state           <= SEND_RESULT_CMD;
                    end
                    else begin
                        dma_cmd_state           <= WAIT_RD_EN;
                    end
                end
                SEND_RESULT_CMD:begin
                    dma_write_valid             <= 1;
                    if(m_axis_dma_write_cmd.valid & m_axis_dma_write_cmd.ready)begin
                        dma_write_valid         <= 0;
                        dma_add                 <= 1;
                        dma_cmd_state           <= SEND_NEXT_RESULT_CMD;
                    end
                    else begin
                        dma_cmd_state           <= SEND_RESULT_CMD;
                    end                    
                end
                SEND_NEXT_RESULT_CMD:begin
                    if(dma_total_cnt > (client_num<<<1))begin
                        dma_cmd_state	        <= END;
                    end      
                    else begin
                        dma_cmd_state           <= SEND_RESULT_CMD;
                    end          
                end                
                END:begin
                    dma_cmd_state               <= IDLE;
                end
            endcase    
        end
    end


    always@(posedge clk)begin
        if(~rstn)begin
            total_recv_num                      <= 1'b0;
        end
        else if(recv_end)begin
            total_recv_num                      <= total_recv_num + 1'b1;
        end
        else begin
            total_recv_num                      <= total_recv_num;
        end
    end    






    always@(posedge clk)begin
        mpi_reduce_done                         <= state == END;
        data_cnt_minus                          <= (client_length >>> 6)-1;
        recv_start_r                            <= recv_start;
        session_fifo_rd_pre                     <= axis_session_data.ready & axis_session_data.valid;
        session_fifo_rd_pre_r                   <= session_fifo_rd_pre;
        mem_fifo_rd_pre                         <= axis_mem_read_data.ready & axis_mem_read_data.valid;
        mem_fifo_rd_pre_r                       <= mem_fifo_rd_pre;
        num_mem_ops_minus_1                     <= (client_length>>>9) -1;
    end


    always@(posedge clk)begin
        if(~rstn)begin
            state                               <= IDLE;
            data_ready                          <= data_ready;
            cnt                                 <= 0;
            mem_flag                            <= 0;
            recv_end                            <= 0;
            mem_awvalid_en                      <= 0;
            clear_wr_data_done                  <= 0;
            mpi_reduce_process                  <= 0;
        end
        else begin
            data_ready                          <= 1'b0;
            send_enable_wr_en                   <= 1'b0;
            recv_end                            <= 1'b0;
            mem_awvalid_en                      <= 1'b0;
            clear_wr_data_done                  <= 1'b0;
            case(state)
                IDLE:begin
                    if(start_cal_flag)begin
                        state                   <= FIRST_DMA_READ_CMD;
                        mpi_reduce_process      <= 1'b1;
                    end
                    else begin
                        state                   <= IDLE;
                    end
                end
                FIRST_DMA_READ_CMD:begin
                    // if(m_axis_dma_read_cmd.valid & m_axis_dma_read_cmd.ready)begin
                        mem_awvalid_en          <= 1'b1;
                        state                   <= FIRST_MEM_WRITE_ADDR;
                    // end 
                    // else begin
                    //     state                   <= FIRST_DMA_READ_CMD;
                    // end 
                end
                FIRST_MEM_WRITE_ADDR:begin
                    if((mem_addr == (client_length - 512)) && m_axis_mem_write.awvalid & m_axis_mem_write.awready)begin
                        state                   <= START_CAL;
                    end
                    else begin
                        state                   <= FIRST_MEM_WRITE_ADDR;
                    end
                end
                START_CAL:begin
                    if(wr_data_done)begin
                        state                   <= START_CAL_R;
                        send_enable_wr_en       <= 1'b1;
                        recv_end                <= 1'b1;
                        clear_wr_data_done      <= 1'b1;
                        mem_flag                <= ~mem_flag;                        
                    end
                    else begin
                        state                   <= START_CAL;
                    end
                end
                START_CAL_R:begin
                    // if(client_num == 1)begin
                    //     state                   <= SEND_WRITE_CMD;
                    // end
                    // else begin
                        state                   <= SEND_READ_CMD;
                    // end
                end                
                SEND_NEXT_CMD:begin
                    // if((total_recv_num == client_num) && wr_data_done)begin
                    //     mem_flag                <= ~mem_flag;
                    //     send_enable_wr_en       <= 1'b1;
                    //     clear_wr_data_done      <= 1'b1;
                    //     state	                <= SEND_WRITE_CMD;
                    // end      
                    // if((total_recv_num == (client_num + 1)) && wr_data_done)begin
                    //     mem_flag                <= ~mem_flag;
                    //     send_enable_wr_en       <= 1'b1;
                    //     clear_wr_data_done      <= 1'b1;
                    //     state	                <= SEND_RESULT_CMD;                        
                    // end
                    if(wr_data_done)begin
                        mem_flag                <= ~mem_flag;
                        send_enable_wr_en       <= 1'b1;
                        clear_wr_data_done      <= 1'b1;
                        state                   <= WAIT_RD_EN;
                    end 
                    else begin
                        state                   <= SEND_NEXT_CMD;
                    end            
                end
                WAIT_RD_EN:begin
                    if(send_enable_rd_en)begin
                        if((total_recv_num >= (client_num + 1)))begin
                            state               <= SEND_RESULT_CMD;
                        end
                        else begin
                            state               <= SEND_READ_CMD;
                        end
                    end
                    else begin
                        state                   <= WAIT_RD_EN;
                    end
                end                
                SEND_READ_CMD:begin
                    // if(m_axis_dma_read_cmd.valid & m_axis_dma_read_cmd.ready)begin
                        mem_awvalid_en          <= 1'b1;
                        state                   <= CAL_START;
                    // end
                    // else begin
                    //     state                   <= SEND_READ_CMD;
                    // end
                end

                CAL_START:begin
                    if(axis_mem_read_data.last & axis_mem_read_data.ready & axis_mem_read_data.valid)begin
                        recv_end                <= 1'b1;
                        state                   <= SEND_NEXT_CMD;
                    end                    
                    else if(((session_fifo_rd_pre && session_fifo_rd_pre_r && (session_data_count >= 6) && mem_fifo_rd_pre && mem_fifo_rd_pre_r && (dma_read_count >= 6)) ||
                    (~session_fifo_rd_pre && ~session_fifo_rd_pre_r && (session_data_count >= 4) && ~mem_fifo_rd_pre && ~mem_fifo_rd_pre_r && (dma_read_count >= 4))) 
                    & ~mem_write_data_fifo_almost_full & ~dma_write_data_fifo_almost_full)begin
                        state                   <= CAL;
                        data_ready              <= 1'b1;
                    end
                    else begin
                        state                   <= CAL_START;
                    end
                end
                CAL:begin
                    data_ready                  <= 1'b1;
                    cnt                         <= cnt + 1'b1;
                    if(cnt == 2)begin
                        cnt                     <= 0;
                        state                   <= CAL_START;
                    end 
                    else begin
                        state                   <= CAL;
                    end
                end

                SEND_RESULT_CMD:begin
                    // if(m_axis_dma_write_cmd.valid & m_axis_dma_write_cmd.ready)begin
                        mem_awvalid_en          <= 1'b1;
                        state                   <= SEND_RESULT;
                    // end
                    // else begin
                    //     state                   <= SEND_RESULT_CMD;
                    // end                    
                end
                SEND_RESULT:begin
                    data_ready                  <= 1'b1;
                    if(axis_cal_data.last)begin
                        data_ready              <= 1'b0;
                        state                   <= SEND_NEXT_RESULT_CMD;
                        recv_end                <= 1'b1;
                    end
                    else begin
                        state                   <= SEND_RESULT;
                    end
                end
                SEND_NEXT_RESULT_CMD:begin
                    if(wr_data_done && m_axis_dma_write_data.last && (total_recv_num > ((client_num<<<1) + 1) ))begin
                        mem_flag                <= ~mem_flag;
                        // send_enable_wr_en       <= 1'b1;
                        clear_wr_data_done      <= 1'b1;
                        state	                <= END;
                    end      
                    else if(wr_data_done & m_axis_dma_write_data.last)begin
                        mem_flag                <= ~mem_flag;
                        send_enable_wr_en       <= 1'b1;
                        clear_wr_data_done      <= 1'b1;
                        state                   <= WAIT_RD_EN;
                    end   
                    else begin
                        state                   <= SEND_NEXT_RESULT_CMD;
                    end          
                end                
                END:begin
                    state                       <= IDLE;
                    mpi_reduce_process          <= 1'b0;
                end
            endcase    
        end
    end

    reg[3:0]                                    start_cnt;

    always@(posedge clk)begin
        if(~rstn)begin
            start_state                         <= IDLE;
            send_start_wr_en                    <= 0;
            start_cnt                           <= 0;
            start_cal_flag                      <= 0;
        end
        else begin
            send_start_wr_en                    <= 1'b0;
            start_cal_flag                      <= 1'b0;
            case(start_state)
                IDLE:begin
                    if(recv_start_r)begin
                        start_state             <= RECV_START_INFO;
                    end
                    else begin
                        start_state             <= IDLE;
                    end
                end
                RECV_START_INFO:begin
                    if(s_axis_tcp_rx_data.valid & s_axis_tcp_rx_data.ready)begin
                        send_start_wr_en        <= 1'b1;
                        start_cnt               <= start_cnt + 1'b1;
                        start_state             <= RECV_NEXT_INFO;
                    end 
                    else begin
                        start_state             <= RECV_START_INFO;
                    end 
                end
                RECV_NEXT_INFO:begin
                    if(start_cnt == client_num)begin
                        start_state             <= END;
                    end
                    else begin
                        start_state             <= RECV_START_INFO;
                    end
                end            
                END:begin
                    start_cal_flag              <= 1'b1;           
                    start_state                 <= IDLE;
                end
            endcase    
        end
    end
 

ila_mpi_cal ila_mpi_cal (
	.clk(clk), // input wire clk


	.probe0(m_axis_mem_write.awvalid), // input wire [0:0]  probe0  
	.probe1(m_axis_mem_write.awready), // input wire [0:0]  probe1 
	.probe2(m_axis_mem_write.awaddr), // input wire [31:0]  probe2 
	.probe3(m_axis_mem_write.wvalid), // input wire [0:0]  probe3 
	.probe4(m_axis_mem_write.wready), // input wire [0:0]  probe4 
	.probe5(m_axis_mem_write.wlast), // input wire [0:0]  probe5 
	.probe6(m_axis_mem_write.wdata[31:0]), // input wire [31:0]  probe6 
	.probe7(s_axis_tcp_rx_data.ready), // input wire [0:0]  probe7 
	.probe8(s_axis_tcp_rx_data.valid), // input wire [0:0]  probe8 
	.probe9(s_axis_tcp_rx_data.last), // input wire [31:0]  probe9 
	.probe10(recv_cnt), // input wire [7:0]  probe10 
	.probe11(state), // input wire [3:0]  probe11 
    .probe12(start_state), // input wire [3:0]  probe12
    .probe13(session_data_count), // input wire [0:0]  probe8
    .probe14(dma_read_count), // input wire [0:0]  probe8 
    .probe15(cal_read_count), // input wire [0:0]  probe8  
    .probe16(dma_wr_read_count), // input wire [0:0]  probe8  
    .probe17(total_recv_num), // input wire [0:0]  probe8 
	.probe18(s_axis_tcp_rx_meta.valid), // input wire [0:0]  probe18 
	.probe19(s_axis_tcp_rx_meta.ready), // input wire [0:0]  probe19 
	.probe20(s_axis_tcp_rx_meta.data[31:0]), // input wire [63:0]  probe20 
	.probe21(s_axis_tcp_rx_data.valid), // input wire [0:0]  probe21 
	.probe22(s_axis_tcp_rx_data.ready), // input wire [0:0]  probe22 
	.probe23(s_axis_tcp_rx_data.data[31:0]) // input wire [63:0]  probe23    
);

endmodule
