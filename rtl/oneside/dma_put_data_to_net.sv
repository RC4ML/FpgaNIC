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

module dma_put_data_to_net( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,

    //DMA Commands
   	axis_mem_cmd.master         axis_dma_read_cmd,
    //DMA Data streams      
	axi_stream.slave            axis_dma_read_data,
	
	//tcp send
    axis_meta.master     		m_axis_tx_metadata,
    axi_stream.master    		m_axis_tx_data,
    axis_meta.slave    			s_axis_tx_status,

	//control reg
	axis_meta.slave             s_axis_put_data_to_net,   //send cmd and data
	axis_meta.slave				s_axis_get_data_cmd, 	//send cmd
	axis_meta.slave				s_axis_put_data_cmd,	//send cmd and data
	input wire 					recv_done,
	input wire[15:0][31:0]		control_reg,
	output wire[7:0][31:0]		status_reg

	
	);	  

///////////////////////dma_ex debug//////////

	axi_stream								axis_tx_data();
	reg [63:0]            					dma_base_addr;                          

	localparam [3:0]		IDLE 			= 4'h0,
							START			= 4'h1,
							READ_CMD		= 4'h2,
							SEND_METADATA   = 4'h3,
							READ_DATA		= 4'h4,
							GET_DATA		= 4'h5,
							SEND_GET_DATA	= 4'h6,
							SEND_CTRL		= 4'h7,
							SEND_CTRL_DATA	= 4'h8,																				
							JUDGE			= 4'h9;	





	reg [3:0]								state;							

	reg [31:0]								data_cnt;
	reg [31:0]								data_cnt_minus;

    reg [31:0]                              current_addr;
	reg [31:0]                              current_length,tmp_length;
	reg [15:0]								current_session_id;

	reg [511:0]								tcp_tx_data;

	reg [31:0]								token_divide_num;
	reg [31:0]								token_mul;
	reg [31:0] 								opposite_addr;
	



	always @(posedge clk)begin
		dma_base_addr						<= {control_reg[1],control_reg[0]};
		token_divide_num					<= control_reg[2];
		token_mul							<= control_reg[3];


		data_cnt_minus						<= (current_length >>>6)-1;
	end

	//////////////send conn cmd ///////////
	reg 									get_data_en;	
	reg [111:0]								get_data_data;	


	always@(posedge clk)begin
		if(~rstn)begin
			get_data_en					<= 1'b0;
		end
		else if(s_axis_get_data_cmd.ready & s_axis_get_data_cmd.valid)begin
			get_data_en					<= 1'b1;
		end
		else if(state == GET_DATA)begin
			get_data_en					<= 1'b0;
		end		
		else begin
			get_data_en					<= get_data_en;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			get_data_data					<= 1'b0;
		end
		else if(s_axis_get_data_cmd.ready & s_axis_get_data_cmd.valid)begin
			get_data_data					<= s_axis_get_data_cmd.data;
		end	
		else begin
			get_data_data					<= get_data_data;
		end
	end

	assign s_axis_get_data_cmd.ready = ~get_data_en;	
////////////////////////////////////////////////////////	
	//////////////cmd buffer ///////////
	reg 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_almostfull;	
	wire 									fifo_cmd_empty;	
	reg  [111:0]							fifo_cmd_wr_data;
	wire [111:0]							fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;
	wire [9:0]								fifo_cmd_count;	


	assign s_axis_put_data_to_net.ready 	= ~fifo_cmd_almostfull;
	assign s_axis_put_data_cmd.ready 		= ~fifo_cmd_almostfull;

	always@(posedge clk)begin
		fifo_cmd_wr_en						<= (s_axis_put_data_to_net.ready & s_axis_put_data_to_net.valid) || (s_axis_put_data_cmd.ready & s_axis_put_data_cmd.valid);
	end

	always@(posedge clk)begin
		if(~rstn)begin
			fifo_cmd_wr_data				<= 0;
		end
		else if(s_axis_put_data_to_net.ready & s_axis_put_data_to_net.valid)begin
			fifo_cmd_wr_data				<= {s_axis_put_data_to_net.data[111:96],s_axis_put_data_to_net.data[63:32],s_axis_put_data_to_net.data[95:64],s_axis_put_data_to_net.data[31:0]};
		end
		else if(s_axis_put_data_cmd.ready & s_axis_put_data_cmd.valid)begin
			fifo_cmd_wr_data				<= s_axis_put_data_cmd.data;
		end
		else begin
			fifo_cmd_wr_data				<= fifo_cmd_wr_data;
		end
	end

	blockram_fifo #( 
		.FIFO_WIDTH      ( 112 ), //64 
		.FIFO_DEPTH_BITS ( 9 )  //determine the size of 16  13
	) inst_cmd_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_cmd_wr_en     ), //or one cycle later...
	.din        (fifo_cmd_wr_data 	),
	.almostfull (fifo_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_cmd_rd_en     ),
	.dout       (fifo_cmd_rd_data   ),
	.valid      (fifo_cmd_rd_valid	),
	.empty      (fifo_cmd_empty     ),
	.count      (fifo_cmd_count   )
	);

////////////////////////////////////////////////////////
//////////////////////////cmd token////////////////////
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

/////////////////////////////////////////////////////////////



	reg 									tx_metadata_valid;	

	assign	axis_dma_read_cmd.address	    = dma_base_addr + current_addr;
	assign	axis_dma_read_cmd.length	    = current_length; 
	assign 	axis_dma_read_cmd.valid			= (state == READ_CMD); 
	
	assign 	m_axis_tx_metadata.data			= {current_length,current_session_id};
	assign 	m_axis_tx_metadata.valid		= (state == GET_DATA) || (state == SEND_CTRL) || (state == SEND_METADATA);

	assign s_axis_tx_status.ready 			= 1;




	always@(posedge clk)begin
		if(~rstn)begin
			tcp_tx_data						<= 1'b0;
		end
		else if(state == GET_DATA)begin
			tcp_tx_data						<= {384'h0,get_data_data,16'h4};
		end	
		else if(state == SEND_CTRL)begin
			tcp_tx_data						<= {432'h0,opposite_addr,tmp_length,16'h5};
		end		
		else begin
			tcp_tx_data						<= tcp_tx_data;
		end
	end
	


	axi_stream 								axis_dma_data();

	assign axis_dma_data.valid 		= ((big_cnt <<< token_mul) > data_cnt) ? axis_dma_read_data.valid : 0;
	assign axis_dma_read_data.ready = ((big_cnt <<< token_mul) > data_cnt) ? axis_dma_data.ready : 0;
	assign axis_dma_data.data 		= axis_dma_read_data.data;
	assign axis_dma_data.last 		= (data_cnt == data_cnt_minus) && axis_dma_read_data.ready && axis_dma_read_data.valid;
	assign axis_dma_data.keep 		= axis_dma_read_data.keep;


	assign m_axis_tx_data.valid		= (axis_tx_data.valid && ((state == READ_DATA) || (state == JUDGE))) || (state == SEND_GET_DATA) || (state == SEND_CTRL_DATA);
	assign m_axis_tx_data.keep		= 64'hffff_ffff_ffff_ffff;
	assign m_axis_tx_data.last		= ((state == SEND_GET_DATA) || (state == SEND_CTRL_DATA)) ? 1 : axis_tx_data.last;
	assign m_axis_tx_data.data		= ((state == READ_DATA) || (state == JUDGE)) ? axis_tx_data.data : tcp_tx_data;
	assign axis_tx_data.ready 		= m_axis_tx_data.ready && ((state == READ_DATA) || (state == JUDGE));



	axis_data_fifo_512_d4096 read_data_slice_fifo (
		.s_axis_aresetn(rstn),          // input wire s_axis_aresetn
		.s_axis_aclk(clk),                // input wire s_axis_aclk
		.s_axis_tvalid(axis_dma_data.valid),            // input wire s_axis_tvalid
		.s_axis_tready(axis_dma_data.ready),            // output wire s_axis_tready
		.s_axis_tdata(axis_dma_data.data),              // input wire [511 : 0] s_axis_tdata
		.s_axis_tkeep(axis_dma_data.keep),              // input wire [63 : 0] s_axis_tkeep
		.s_axis_tlast(axis_dma_data.last),              // input wire s_axis_tlast
		.m_axis_tvalid(axis_tx_data.valid),            // output wire m_axis_tvalid
		.m_axis_tready(axis_tx_data.ready),            // input wire m_axis_tready
		.m_axis_tdata(axis_tx_data.data),              // output wire [511 : 0] m_axis_tdata
		.m_axis_tkeep(axis_tx_data.keep),              // output wire [63 : 0] m_axis_tkeep
		.m_axis_tlast(axis_tx_data.last),              // output wire m_axis_tlast
		.axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
		.axis_rd_data_count()  // output wire [31 : 0] axis_rd_data_count
	  );




	always @(posedge clk)begin
		if(~rstn)begin
			data_cnt 						<= 1'b0;
		end
		else if(axis_dma_data.last)begin
			data_cnt						<= 1'b0;
		end
		else if (axis_dma_read_data.ready & axis_dma_read_data.valid)begin
			data_cnt						<= data_cnt + 1'b1;
		end
		else begin
			data_cnt						<= data_cnt;
		end		
	end


	always @(posedge clk)begin
		if(~rstn)begin
			state						<= IDLE;
			token_en					<= 0;
		end
		else begin
			fifo_cmd_rd_en				<= 1'b0;
			case(state)				
				IDLE:begin
					if(get_data_en)begin
						current_length		<= 32'h40;
						current_session_id	<= get_data_data[111:96];
						state				<= GET_DATA;
					end
					else if(~fifo_cmd_empty)begin
						fifo_cmd_rd_en	<= 1'b1;
						state			<= START;
					end
					else begin
						state			<= IDLE;
					end
				end
				GET_DATA:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						state				<= SEND_GET_DATA;
					end
					else begin
						state				<= GET_DATA;
					end				
				end
				SEND_GET_DATA:begin
					if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
						state				<= IDLE;
					end						
					else begin
						state				<= SEND_GET_DATA;
					end
				end
				START:begin
					if(fifo_cmd_rd_valid)begin
						state           	<= SEND_CTRL;
						current_addr		<= fifo_cmd_rd_data[95:64];
						opposite_addr		<= fifo_cmd_rd_data[63:32];
						tmp_length			<= fifo_cmd_rd_data[31:0];
						current_length		<= 32'h40;
						current_session_id	<= fifo_cmd_rd_data[111:96];
					end
					else begin
						state			<= START;
					end
				end
				SEND_CTRL:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						state				<= SEND_CTRL_DATA;
					end
					else begin
						state				<= SEND_CTRL;
					end
				end
				SEND_CTRL_DATA:begin
					if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
						state				<= READ_CMD;
						current_length		<= tmp_length;
					end						
					else begin
						state				<= SEND_CTRL_DATA;
					end
				end				
				READ_CMD:begin
					if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
						token_en		<= 1;
						state			<= SEND_METADATA;
					end
					else begin
						state			<= READ_CMD;
					end
				end
				SEND_METADATA:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						state				<= READ_DATA;
					end
					else begin
						state				<= SEND_METADATA;
					end				
				end
				READ_DATA:begin
					if(axis_dma_data.last)begin
						token_en		<= 0;
						state			<= JUDGE;
					end	
					else begin
						state			<= READ_DATA;
					end
                end
				JUDGE:begin
					if(axis_tx_data.last & m_axis_tx_data.ready & m_axis_tx_data.valid)begin
						state			<= IDLE;
					end
					else begin
						state			<= JUDGE;
					end
				end
			endcase
		end
	end


///////////////////////////////////debug





	reg 									wr_th_en;
	reg [31:0]								wr_th_sum;
	reg [31:0]								wr_data_cnt;
	reg [31:0]								wr_length_minus;

	

	always@(posedge clk)begin
		wr_length_minus							<= control_reg[5] -1;
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_th_en						<= 1'b0;
		end  
		else if(recv_done)begin
			wr_th_en						<= 1'b0;
		end
		else if(s_axis_get_data_cmd.ready & s_axis_get_data_cmd.valid)begin
			wr_th_en						<= 1'b1;
		end		
		else begin
			wr_th_en						<= wr_th_en;
		end
	end

	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		wr_data_cnt						<= 1'b0;
	// 	end  
	// 	else if(axis_dma_read_data.ready & axis_dma_read_data.valid)begin
	// 		wr_data_cnt						<= wr_data_cnt + 1'b1;
	// 	end		
	// 	else begin
	// 		wr_data_cnt						<= wr_data_cnt;
	// 	end
	// end
	


	always@(posedge clk)begin
		if(~rstn)begin
			wr_th_sum						<= 32'b0;
		end 
		else if(wr_th_en)begin
			wr_th_sum						<= wr_th_sum + 1'b1;
		end
		else begin
			wr_th_sum						<= wr_th_sum;
		end
	end

	assign status_reg[0]					= wr_th_sum;


	ila_oneside_put ila_oneside_put (
		.clk(clk), // input wire clk
	
	
		.probe0(axis_dma_read_cmd.valid), // input wire [0:0]  probe0  
		.probe1(axis_dma_read_cmd.ready), // input wire [0:0]  probe1 
		.probe2(axis_dma_read_cmd.address), // input wire [63:0]  probe2 
		.probe3(axis_dma_read_cmd.length), // input wire [31:0]  probe3 
		.probe4(axis_dma_data.valid), // input wire [0:0]  probe4 
		.probe5(axis_dma_data.ready), // input wire [0:0]  probe5 
		.probe6(axis_dma_data.last), // input wire [0:0]  probe6 
		.probe7(axis_dma_data.data[31:0]), // input wire [31:0]  probe7 
		.probe8(s_axis_put_data_to_net.valid), // input wire [0:0]  probe8 
		.probe9(s_axis_put_data_to_net.ready), // input wire [0:0]  probe9 
		.probe10(s_axis_get_data_cmd.valid), // input wire [0:0]  probe10 
		.probe11(s_axis_get_data_cmd.ready), // input wire [0:0]  probe11 
		.probe12(s_axis_put_data_cmd.valid), // input wire [0:0]  probe12 
		.probe13(s_axis_put_data_cmd.ready), // input wire [0:0]  probe13 
		.probe14(state) // input wire [3:0]  probe14
	);


endmodule
