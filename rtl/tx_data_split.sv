`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2020/07/31 11:06:25
// Design Name: 
// Module Name: tx_data_split
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


module tx_data_split(
    input               clk,
    input               rstn,

    //input tx_data
    axis_meta.slave     s_axis_tx_metadata,
    axi_stream.slave    s_axis_tx_data,
    axis_meta.master    m_axis_tx_status,
    //output tx_data splited
    axis_meta.master    m_axis_tx_metadata,
    axi_stream.master   m_axis_tx_data,
    axis_meta.slave     s_axis_tx_status,

    input wire [31:0]   mtu


    );

    wire m_axis_tlast;
    wire s_axis_tready;
    wire s_axis_tvalid;
    wire m_axis_tready;
    wire m_axis_tvalid;


    axis_data_fifo_512_cc axis_data_fifo_512_cc_inst (
        .s_axis_aresetn(rstn),  // input wire s_axis_aresetn
        .s_axis_aclk(clk),        // input wire s_axis_aclk
        .s_axis_tvalid(s_axis_tvalid),    // input wire s_axis_tvalid
        .s_axis_tready(s_axis_tready),    // output wire s_axis_tready
        .s_axis_tdata(s_axis_tx_data.data),      // input wire [63 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_tx_data.keep),      // input wire [7 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_tx_data.last),      // input wire s_axis_tlast
        .m_axis_aclk(clk),        // input wire m_axis_aclk
        .m_axis_tvalid(m_axis_tvalid),    // output wire m_axis_tvalid
        .m_axis_tready(m_axis_tready),    // input wire m_axis_tready
        .m_axis_tdata(m_axis_tx_data.data),      // output wire [63 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_tx_data.keep),      // output wire [7 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_tlast)      // output wire m_axis_tlast
      );



    localparam [15:0]   MAX_LENGTH      = 32'd1408;

    // reg [31:0]      MAX_LENGTH;
    // always@(posedge clk)begin
    //     if(mtu == 0)begin
    //         MAX_LENGTH  <= 32'd1408;
    //     end
    //     else begin
    //         MAX_LENGTH  <= mtu;
    //     end
    // end
    // wire [15:0]   MAX_LENGTH;
    // vio_0 vio_0 (
    // .clk(clk),                // input wire clk
    // .probe_out0(MAX_LENGTH)  // output wire [15 : 0] probe_out0
    // );
    localparam [3:0]    IDLE            = 9'h0,  
                        META            = 9'h1, 
                        SPLIT_META      = 9'h2,
                        LAST_META       = 9'h3,
                        SPLIT_STS       = 9'h4,
                        LAST_STS        = 9'h5,                          
                        STS             = 9'h6,
                        WAIT_DATA       = 9'h7,
                        LAST_DATA       = 9'h8;
                        
    reg [3:0]           cstate,nstate;

    reg [31:0]          tx_metadata; 
    reg [15:0]          tx_session;
    reg [31:0]          tx_data_length,tx_data_length_data;  
    reg [63:0]          s_tx_sts_data,m_tx_sts_data;    

    reg [7:0]           pkg_cnt;
    reg [7:0]           pkg_cycle;
    
    always @(posedge clk) begin
        if (~rstn) begin
            pkg_cnt                     <=  8'b0;
        end
        else if (m_axis_tx_data.valid & m_axis_tx_data.ready) begin
            pkg_cnt                     <= pkg_cnt + 8'b1;
            if (m_axis_tx_data.last)begin
                pkg_cnt                 <= 8'b0;
            end    
        end        
    end

    always@(posedge clk)begin
        pkg_cycle                       <= (MAX_LENGTH>>>6)-1;
    end

    
    always @(posedge clk) begin
        if (~rstn) begin
            m_tx_sts_data               <=  64'b0;
        end
        else if(m_axis_tx_status.ready & m_axis_tx_status.valid)begin
            m_tx_sts_data               <=  64'b0;
        end
        // else if (s_tx_sts_data == 0) begin
        //     m_tx_sts_data               <= m_tx_sts_data;   
        // end    
        else begin
            m_tx_sts_data               <= s_tx_sts_data;
        end
    end
    ///////////////////////////////////////////
    reg [15:0]          meta_count ;
    reg [15:0]          data_count ;
    reg                 overflow_flag;
    reg                 wait_data_r,last_data_r;

    always@(posedge clk)begin
        wait_data_r                     <= cstate == WAIT_DATA;
        last_data_r                     <= cstate == LAST_DATA;
    end

    always@(posedge clk)begin
        if(~rstn)begin
            meta_count                  <= 0;
        end
        else if(((cstate == WAIT_DATA) && (~wait_data_r)) || ((cstate == LAST_DATA) && (~last_data_r)))begin
            meta_count                  <= meta_count + 1'b1;
        end
        else begin
            meta_count                  <= meta_count;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            data_count                  <= 0;
        end
        else if(m_axis_tx_data.last)begin
            data_count                  <= data_count + 1'b1;
        end
        else begin
            data_count                  <= data_count;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            overflow_flag               <= 0;
        end
        else if((meta_count == 16'hffff)&&(((cstate == WAIT_DATA) && (~wait_data_r)) || ((cstate == LAST_DATA) && (~last_data_r))))begin
            overflow_flag               <= 1'b1;
        end
        else if((data_count == 16'hffff)&&m_axis_tx_data.last)begin
            overflow_flag               <= 1'b0;
        end
        else begin
            overflow_flag               <= overflow_flag;
        end
    end
//////////////////////////////////////////////////////////

    assign s_axis_tx_metadata.ready     = (cstate == IDLE);
    assign m_axis_tx_metadata.valid     = (cstate == SPLIT_META) || (cstate == LAST_META);
    assign m_axis_tx_metadata.data      = tx_metadata;
//    assign s_axis_tx_metadata.ready     = m_axis_tx_metadata.ready;

    assign s_axis_tx_data.ready         = s_axis_tready ;
    assign s_axis_tvalid                = s_axis_tx_data.valid;
    
    assign m_axis_tx_data.valid         = m_axis_tvalid && ((meta_count>data_count) || overflow_flag);
    assign m_axis_tready                = m_axis_tx_data.ready && ((meta_count>data_count) || overflow_flag);    
    
    
    assign m_axis_tx_data.last          = (m_axis_tlast && (m_axis_tx_data.valid & m_axis_tx_data.ready)) || ((pkg_cnt == pkg_cycle) && (m_axis_tx_data.valid & m_axis_tx_data.ready));
//    assign m_axis_tready                = m_axis_tx_data.ready;// & cstate[]

    assign s_axis_tx_status.ready       = (cstate == SPLIT_STS) || (cstate == LAST_STS);
    assign m_axis_tx_status.valid       = cstate == STS;
    assign m_axis_tx_status.data        = m_tx_sts_data;


    always @(posedge clk) begin
        if (~rstn) begin
            cstate                      <=  IDLE;
        end
        else begin
            cstate                      <=  nstate;
        end        
    end


    always @(*)begin
        nstate                          = IDLE;
        case(cstate)
            IDLE:begin
                if(s_axis_tx_metadata.valid & s_axis_tx_metadata.ready)begin
                    nstate              = META;
                end
                else begin
                    nstate              = IDLE;
                end
            end
            META:begin
                if(tx_data_length > MAX_LENGTH) begin
                    nstate              = SPLIT_META;
                end
                else begin
                    nstate              = LAST_META;
                end
            end
            SPLIT_META:begin
                if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
                    nstate              = SPLIT_STS;
                end
                else begin
                    nstate              = SPLIT_META;
                end
            end            
            SPLIT_STS:begin
                if(s_axis_tx_status.ready & s_axis_tx_status.valid)begin
                    if(s_axis_tx_status.data[63])begin
                        nstate          = META;
                    end
                    else if(s_axis_tx_status.data[62])begin
                        nstate          = STS;
                    end
                    else begin
                        nstate          = WAIT_DATA;
                    end                    
                end
                else begin
                    nstate              = SPLIT_STS;
                end
            end
            WAIT_DATA:begin
                if( (data_count >= (meta_count-1)) || (meta_count == 0) )begin
                    nstate              = META;
                end
                else begin
                    nstate              = WAIT_DATA;
                end
            end            
            LAST_META:begin
                if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
                    nstate              = LAST_STS;
                end
                else begin
                    nstate              = LAST_META;
                end                
            end            
            LAST_STS:begin
                if(s_axis_tx_status.ready & s_axis_tx_status.valid)begin
                    if(s_axis_tx_status.data[63])begin
                        nstate          = META;
                    end
                    else if(s_axis_tx_status.data[62])begin
                        nstate          = STS;
                    end
                    else begin
                        nstate          = LAST_DATA;
                    end  
                end  
                else begin
                    nstate              = LAST_STS;
                end           
            end
            LAST_DATA:begin
                if(m_axis_tlast && (m_axis_tx_data.valid & m_axis_tx_data.ready) )begin
                    nstate              = STS;
                end
                else begin
                    nstate              = LAST_DATA;
                end
            end             
            STS:begin
                if(m_axis_tx_status.ready & m_axis_tx_status.valid)begin
                    nstate              = IDLE;
                end
                else begin
                    nstate              = STS;
                end
            end
        endcase
    end






    reg                                 length_minus_flag,last_length_minus_flag;
    
    always @(posedge clk)begin
        length_minus_flag               <= (cstate == WAIT_DATA);
        last_length_minus_flag          <= (cstate == LAST_DATA);
    end


    always @(posedge clk)begin
        if(~rstn)begin
            tx_data_length              <= 0;
            tx_session                  <= 0;
        end
        else if(s_axis_tx_metadata.valid && s_axis_tx_metadata.ready)begin
            tx_data_length              <= s_axis_tx_metadata.data[47:16];
            tx_session                  <= s_axis_tx_metadata.data[15:0];  
        end
        else if((tx_data_length > MAX_LENGTH) && (cstate == WAIT_DATA) && (~length_minus_flag))begin
            tx_data_length              <= tx_data_length -  MAX_LENGTH;
            tx_session                  <= tx_session;          
        end   
        else if((cstate == LAST_DATA) && (~last_length_minus_flag))begin
            tx_data_length              <= 0;
            tx_session                  <= tx_session;             
        end
        else begin
            tx_data_length              <= tx_data_length;
            tx_session                  <= tx_session;             
        end
    end

	//////////////cmd buffer ///////////
	reg 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_almostfull;	
	wire 									fifo_cmd_empty;	
	wire [31:0]								fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;	

	assign fifo_cmd_wr_en					= s_axis_tx_metadata.valid && s_axis_tx_metadata.ready;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 32 ), //64 
		.FIFO_DEPTH_BITS ( 9 )  //determine the size of 16  13
	) inst_a_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_cmd_wr_en     ), //or one cycle later...
	.din        (s_axis_tx_metadata.data[47:16] ),
	.almostfull (fifo_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_cmd_rd_en     ),
	.dout       (fifo_cmd_rd_data   ),
	.valid      (fifo_cmd_rd_valid	),
	.empty      (fifo_cmd_empty     ),
	.count      (   )
	);

////////////////////////////////////////////////////////
    reg                             read_flag;

    always @(posedge clk)begin
        if(~rstn)begin
            fifo_cmd_rd_en          <= 0;
            read_flag               <= 0;
        end
        else if((~read_flag) && (~fifo_cmd_empty))begin
            fifo_cmd_rd_en          <= 1;
            read_flag               <= 1;           
        end
        else if(m_axis_tlast && m_axis_tx_data.valid && m_axis_tx_data.ready)begin
            fifo_cmd_rd_en          <= 0;  
            read_flag               <= 1'b0; 
        end
        else begin
            fifo_cmd_rd_en          <= 0;
            read_flag               <= read_flag;              
        end
    end   

    always @(posedge clk)begin
        if(~rstn)begin
            tx_data_length_data         <= 0;
        end
        else if(fifo_cmd_rd_valid)begin
            tx_data_length_data         <= fifo_cmd_rd_data;  
        end
        else if((tx_data_length_data > MAX_LENGTH) && (pkg_cnt == pkg_cycle) && (m_axis_tx_data.valid & m_axis_tx_data.ready))begin
            tx_data_length_data         <= tx_data_length_data -  MAX_LENGTH;         
        end   
        else if((tx_data_length_data <= MAX_LENGTH) && m_axis_tlast && m_axis_tx_data.valid && m_axis_tx_data.ready)begin
            tx_data_length_data         <= 0;
        end
        else begin
            tx_data_length_data         <= tx_data_length_data;             
        end
    end    

    

    always @(posedge clk)begin
        if(~rstn)begin
            tx_metadata                 <= 0;
        end
        else if(tx_data_length > MAX_LENGTH && (cstate == META))begin
            tx_metadata                 <= {MAX_LENGTH,tx_session};          
        end   
        else if(cstate == META)begin
            tx_metadata                 <= {tx_data_length[15:0],tx_session};            
        end
        else begin
            tx_metadata                 <= tx_metadata;
        end
    end    

    always @(posedge clk)begin
        if(~rstn)begin
            s_tx_sts_data               <= 0;
        end
        else if(s_axis_tx_status.ready & s_axis_tx_status.valid)begin
            s_tx_sts_data               <= s_axis_tx_status.data;          
        end   
        else begin
            s_tx_sts_data               <= s_tx_sts_data;
        end
    end



ila_split probe_ila_split(
.clk(clk),

.probe0(s_axis_tx_metadata.valid), // input wire [1:0]
.probe1(s_axis_tx_metadata.ready), // input wire [1:0]
.probe2(s_axis_tx_metadata.data), // input wire [32:0]
.probe3(s_axis_tx_data.valid), // input wire [1:0]
.probe4(s_axis_tx_data.ready), // input wire [1:0]
.probe5(s_axis_tx_data.last), // input wire [1:0]
.probe6(s_axis_tx_data.data[31:0]), // input wire [64:0]
.probe7(fifo_cmd_rd_en), // input wire [1:0]
.probe8(read_flag), // input wire [1:0]
.probe9(fifo_cmd_rd_data), // input wire [64:0]
.probe10(m_axis_tx_metadata.valid), // input wire [1:0]
.probe11(m_axis_tx_metadata.ready), // input wire [1:0]
.probe12(m_axis_tx_metadata.data), // input wire [32:0]
.probe13(m_axis_tx_data.valid), // input wire [1:0]
.probe14(m_axis_tx_data.ready), // input wire [1:0]
.probe15(m_axis_tx_data.last), // input wire [1:0]
.probe16(m_axis_tx_data.data[31:0]), // input wire [64:0]
.probe17(fifo_cmd_rd_valid), // input wire [1:0]
.probe18(fifo_cmd_empty), // input wire [1:0]
.probe19({m_axis_tlast,meta_count,data_count}), // input wire [64:0]
.probe20(cstate), // input wire [7:0]  probe20 
.probe21(tx_data_length), // input wire [15:0]  probe21
.probe22(tx_data_length_data), // input wire [15:0]  probe21
.probe23(s_axis_tx_status.ready), // input wire [0:0]  probe23 
.probe24(s_axis_tx_status.valid), // input wire [0:0]  probe24 
.probe25(s_axis_tx_status.data) // input wire [63:0]  probe25

);


endmodule

