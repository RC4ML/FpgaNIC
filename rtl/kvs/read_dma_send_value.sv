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

module read_dma_send_value( 

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
	// axis_meta.slave             s_axis_conn_send,   //send conn req
	// axis_meta.slave				s_axis_conn_ack,    //send conn ack req 

    // axis_meta.slave    			s_axis_tcp_send_write_cnt,   //dont send to net
	// axis_meta.slave    			s_axis_send_read_cnt,		//send to net
	// axis_meta.slave    			axis_tcp_recv_read_cnt,		//dont send to net
	// axis_meta.slave				s_axis_cmd, 				//send
	input wire[15:0][31:0]		control_reg,
	output wire[7:0][31:0]		status_reg

	
	);
	

   


///////////////////////dma_ex debug//////////

	axi_stream								axis_tx_data();
	reg [63:0]            					dma_base_addr,dma_addr;   
	reg [31:0]								dma_length;              
	reg [31:0]            					dma_real_length,dma_align_length;
	reg [31:0]								dma_info_cl_minus;	                    

	localparam [3:0]		IDLE 			= 4'h0,
							START			= 4'h2,
							READ_CMD		= 4'h3,
							READ_DATA		= 4'h4,
							READ_INFO		= 4'h5,
							SEND_METADATA	= 4'h6,
							SEND_DATA		= 4'h7;



	reg [3:0]								state,s_state;							

	reg [31:0]								rd_start_r,rd_start_rr;
	reg [31:0]								data_cnt,tx_data_cnt;
	reg [31:0]								data_cnt_minus,tx_data_cnt_minus;

	reg [15:0]                              current_length,current_session_id;
	reg [15:0][31:0]						tcp_tx_info;
	reg [7:0]								info_cnt,info_index;
	reg [8:0]								delay_count;

	



	always @(posedge clk)begin
		rd_start_r							<= control_reg[5];
		rd_start_rr							<= rd_start_r;
		dma_base_addr						<= {control_reg[1],control_reg[0]};
		dma_real_length						<= control_reg[2];
		// dma_align_length					<= control_reg[3];
		dma_info_cl_minus					<= control_reg[3][7:0];
		delay_count							<= control_reg[4][8:0];
	end




	//////////////delay buffer ///////////
	reg [511:0]								fifo_delay_wr_en;
	reg 									fifo_delay_rd_en;			
	wire 									fifo_delay_almostfull;	
	wire 									fifo_delay_empty;	
	reg [95:0]								fifo_delay_wr_data;
	wire [95:0]								fifo_delay_rd_data;
	wire 									fifo_delay_rd_valid;
	wire [9:0]								fifo_delay_count;	

	always@(posedge clk)begin
		fifo_delay_wr_data						<= {dma_real_length,dma_base_addr};
		fifo_delay_wr_en[0]						<= (rd_start_r != rd_start_rr);
		fifo_delay_wr_en[511:1]					<= fifo_delay_wr_en[510:0];
		fifo_delay_rd_en						<= fifo_delay_wr_en[delay_count];
	end

	blockram_fifo #( 
		.FIFO_WIDTH      ( 96 ), //64 
		.FIFO_DEPTH_BITS ( 10 )  //determine the size of 16  13
	) inst_delay_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_delay_wr_en[0] ), //or one cycle later...
	.din        (fifo_delay_wr_data ),
	.almostfull (fifo_delay_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_delay_rd_en     ),
	.dout       (fifo_delay_rd_data   ),
	.valid      (fifo_delay_rd_valid	),
	.empty      (fifo_delay_empty     ),
	.count      (fifo_delay_count   )
	);

	//////////////dma cmd buffer ///////////
	reg 									fifo_dma_cmd_wr_en;
	reg 									fifo_dma_cmd_rd_en;			
	wire 									fifo_dma_cmd_almostfull;	
	wire 									fifo_dma_cmd_empty;	
	reg [95:0]								fifo_dma_cmd_wr_data;
	wire [95:0]								fifo_dma_cmd_rd_data;
	wire 									fifo_dma_cmd_rd_valid;
	wire [9:0]								fifo_dma_cmd_count;	

	always@(posedge clk)begin
		fifo_dma_cmd_wr_data					<= fifo_delay_rd_data;
		fifo_dma_cmd_wr_en						<= fifo_delay_rd_valid;
		
	end

	blockram_fifo #( 
		.FIFO_WIDTH      ( 96 ), //64 
		.FIFO_DEPTH_BITS ( 10 )  //determine the size of 16  13
	) inst_dma_cmd_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_dma_cmd_wr_en ), //or one cycle later...
	.din        (fifo_dma_cmd_wr_data ),
	.almostfull (fifo_dma_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_dma_cmd_rd_en     ),
	.dout       (fifo_dma_cmd_rd_data   ),
	.valid      (fifo_dma_cmd_rd_valid	),
	.empty      (fifo_dma_cmd_empty     ),
	.count      (fifo_dma_cmd_count   )
	);

	//////////////info buffer ///////////
	wire 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_almostfull;	
	wire 									fifo_cmd_empty;	
	wire [511:0]							fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;
	wire [9:0]								fifo_cmd_count;	

	assign fifo_cmd_wr_en					= axis_dma_read_data.ready & axis_dma_read_data.valid & (state == READ_INFO);

	blockram_fifo #( 
		.FIFO_WIDTH      ( 512 ), //64 
		.FIFO_DEPTH_BITS ( 9 )  //determine the size of 16  13
	) inst_cmd_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_cmd_wr_en     ), //or one cycle later...
	.din        (axis_dma_read_data.data ),
	.almostfull (fifo_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_cmd_rd_en     ),
	.dout       (fifo_cmd_rd_data   ),
	.valid      (fifo_cmd_rd_valid	),
	.empty      (fifo_cmd_empty     ),
	.count      (fifo_cmd_count   )
	);

////////////////////////////////////////////////////////	

	assign	axis_dma_read_cmd.address	    = dma_addr;
	assign	axis_dma_read_cmd.length	    = dma_length; 
	assign 	axis_dma_read_cmd.valid			= (state == READ_CMD); 
	
	assign 	m_axis_tx_metadata.data			= {16'b0,current_length,current_session_id};
	assign 	m_axis_tx_metadata.valid		= (s_state == SEND_METADATA);

	assign s_axis_tx_status.ready 			= 1;
	


	axi_stream 								axis_dma_data();
	reg [31:0]								dma_info_count;

	assign axis_dma_data.valid 		= axis_dma_read_data.valid && (state == READ_DATA);
	assign axis_dma_read_data.ready = (axis_dma_data.ready && (state == READ_DATA)) || (state == READ_INFO);
	assign axis_dma_data.data 		= axis_dma_read_data.data;
	assign axis_dma_data.last 		= (data_cnt == data_cnt_minus) && axis_dma_read_data.ready && axis_dma_read_data.valid;
	assign axis_dma_data.keep 		= axis_dma_read_data.keep;


	assign m_axis_tx_data.valid		= axis_tx_data.valid && (s_state == SEND_DATA);
	assign m_axis_tx_data.keep		= 64'hffff_ffff_ffff_ffff ;
	assign m_axis_tx_data.last		= (tx_data_cnt == tx_data_cnt_minus) &&m_axis_tx_data.ready & m_axis_tx_data.valid;
	assign m_axis_tx_data.data		= axis_tx_data.data;
	assign axis_tx_data.ready 		= m_axis_tx_data.ready && (s_state == SEND_DATA);



	axis_data_fifo_512_d1024 read_data_slice_fifo (
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
		.axis_rd_data_count(dma_info_count)  // output wire [31 : 0] axis_rd_data_count
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
			state							<= IDLE;
			info_cnt						<= 0;
			fifo_dma_cmd_rd_en				<= 1'b0;
			data_cnt_minus					<= 0;
			dma_addr						<= 0;
			dma_length						<= 0;
		end
		else begin
			fifo_dma_cmd_rd_en				<= 1'b0;
			case(state)				
				IDLE:begin
					if(~fifo_dma_cmd_empty)begin
						fifo_dma_cmd_rd_en	<= 1'b1;
						state				<= START;
					end
					else begin
						state				<= IDLE;
					end
				end
				START:begin
					if(fifo_dma_cmd_rd_valid)begin
						state           	<= READ_CMD;
						data_cnt_minus		<= (fifo_dma_cmd_rd_data[95:64]>>6) - 1;
						dma_addr			<= fifo_dma_cmd_rd_data[63:0];
						dma_length			<= fifo_dma_cmd_rd_data[95:64];
					end
					else begin
						state				<= START;
					end
				end				
				READ_CMD:begin
					if(axis_dma_read_cmd.valid & axis_dma_read_cmd.ready)begin
						state           	<= READ_INFO;
					end
					else begin
						state				<= READ_CMD;
					end
				end
				READ_INFO:begin
					if(axis_dma_read_data.valid & axis_dma_read_data.ready)begin
						info_cnt 			<= info_cnt + 1;
						if(info_cnt == (dma_info_cl_minus))begin
							info_cnt		<= 0;
							state			<= READ_DATA;
						end
						else begin
							state			<= READ_INFO;
						end
					end
					else begin
						state				<= READ_INFO;
					end
				end
				READ_DATA:begin
					if(axis_dma_data.last)begin
						state				<= IDLE;
					end
					else begin
						state				<= READ_DATA;
					end
				end
			endcase
		end
	end


	always @(posedge clk)begin
		if(~rstn)begin
			tx_data_cnt 						<= 1'b0;
		end
		else if(m_axis_tx_data.last)begin
			tx_data_cnt						<= 1'b0;
		end
		else if (m_axis_tx_data.ready & m_axis_tx_data.valid)begin
			tx_data_cnt						<= tx_data_cnt + 1'b1;
		end
		else begin
			tx_data_cnt						<= tx_data_cnt;
		end		
	end


	always @(posedge clk)begin
		if(~rstn)begin
			s_state							<= IDLE;
			info_index						<= 0;
			fifo_cmd_rd_en					<= 1'b0;
			tcp_tx_info						<= 0;
			current_length					<= 0;
			current_session_id				<= 0;
			tx_data_cnt_minus				<= 0;
		end
		else begin
			fifo_cmd_rd_en					<= 1'b0;
			case(s_state)				
				IDLE:begin
					if(~fifo_cmd_empty)begin
						fifo_cmd_rd_en		<= 1'b1;
						s_state				<= START;
					end
					else begin
						s_state				<= IDLE;
					end
				end
				START:begin
					if(fifo_cmd_rd_valid)begin
						s_state           	<= SEND_METADATA;
						tcp_tx_info			<= fifo_cmd_rd_data;
						current_length		<= fifo_cmd_rd_data[31:16];
						current_session_id	<= fifo_cmd_rd_data[15:0];
						tx_data_cnt_minus	<= (fifo_cmd_rd_data[31:16]>>6)-1;
						if(fifo_cmd_rd_data[31:16]<16'h40)begin
							s_state         <= IDLE;
						end
					end
					else begin
						s_state				<= START;
					end
				end				
				SEND_METADATA:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						s_state				<= SEND_DATA;
					end
					else begin
						s_state				<= SEND_METADATA;
					end
				end
				SEND_DATA:begin
					if(m_axis_tx_data.last)begin
						if(info_index == 4'hf)begin
							info_index			<= 0;
							s_state				<= IDLE;
						end
						else if(tcp_tx_info[info_index+1][31:16]>=16'h40)begin
							info_index			<= info_index + 1;
							current_length		<= tcp_tx_info[info_index+1][31:16];
							current_session_id	<= tcp_tx_info[info_index+1][15:0];	
							tx_data_cnt_minus	<= (tcp_tx_info[info_index+1][31:16]>>6)-1;
							s_state				<= SEND_METADATA;						
						end
						else begin
							s_state				<= IDLE;
						end						
					end						
					else begin
						s_state					<= SEND_DATA;
					end
				end				
			endcase
		end
	end

//////////////////////////////////////debug//////////////////////////////

	reg 									wr_th_en;
	reg [31:0]								wr_th_sum;
	reg [31:0]								wr_data_cnt;
	reg [31:0]								wr_length_minus,tcp_length_minus;

	

	always@(posedge clk)begin
		wr_length_minus							<= (control_reg[2]>>6) -1;
		tcp_length_minus						<= (control_reg[6]>>6) -1;
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_th_en						<= 1'b0;
		end  
		else if(wr_data_cnt == wr_length_minus)begin
			wr_th_en						<= 1'b0;
		end
		else if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
			wr_th_en						<= 1'b1;
		end		
		else begin
			wr_th_en						<= wr_th_en;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_data_cnt						<= 1'b0;
		end  
		else if(axis_dma_read_data.ready & axis_dma_read_data.valid)begin
			wr_data_cnt						<= wr_data_cnt + 1'b1;
		end		
		else begin
			wr_data_cnt						<= wr_data_cnt;
		end
	end
	


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

	////////////////////////////////////////////////////////////////
	reg 									wr_tcp_en;
	reg [31:0]								wr_tcp_sum;
	reg [31:0]								wr_tcp_data_cnt;	
	reg [31:0]								wr_tcp_valid_sum;
	reg [31:0]								wr_tcp_ready_sum;

	always@(posedge clk)begin
		if(~rstn)begin
			wr_tcp_en						<= 1'b0;
		end  
		else if(wr_tcp_data_cnt == tcp_length_minus)begin
			wr_tcp_en						<= 1'b0;
		end
		else if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
			wr_tcp_en						<= 1'b1;
		end		
		else begin
			wr_tcp_en						<= wr_tcp_en;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_tcp_data_cnt						<= 1'b0;
		end  
		else if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
			wr_tcp_data_cnt						<= wr_tcp_data_cnt + 1'b1;
		end		
		else begin
			wr_tcp_data_cnt						<= wr_tcp_data_cnt;
		end
	end
	


	always@(posedge clk)begin
		if(~rstn)begin
			wr_tcp_sum						<= 32'b0;
		end 
		else if(wr_tcp_en)begin
			wr_tcp_sum						<= wr_tcp_sum + 1'b1;
		end
		else begin
			wr_tcp_sum						<= wr_tcp_sum;
		end
	end

	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		wr_tcp_valid_sum				<= 32'b0;
	// 	end 
	// 	else if(wr_tcp_en && axis_dma_read_data.valid && (~axis_dma_read_data.ready))begin
	// 		wr_tcp_valid_sum				<= wr_tcp_valid_sum + 1'b1;
	// 	end
	// 	else begin
	// 		wr_tcp_valid_sum				<= wr_tcp_valid_sum;
	// 	end
	// end

	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		wr_tcp_ready_sum				<= 32'b0;
	// 	end 
	// 	else if(wr_tcp_en && (~axis_dma_read_data.valid) && axis_dma_read_data.ready)begin
	// 		wr_tcp_ready_sum				<= wr_tcp_ready_sum + 1'b1;
	// 	end
	// 	else begin
	// 		wr_tcp_ready_sum				<= wr_tcp_ready_sum;
	// 	end
	// end

	assign status_reg[1]					= wr_tcp_sum;
	assign status_reg[2]					= dma_info_count;
	assign status_reg[3]					= fifo_dma_cmd_count;

	// ////////////////////////////////////////////////////////////////
	// reg [1:0]								cmd_delay_en;
	// reg [31:0]								cmd_delay_sum;	

	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		cmd_delay_en						<= 1'b0;
	// 	end  
	// 	else if(cmd_delay_en == 2)begin
	// 		cmd_delay_en						<= cmd_delay_en;
	// 	end
	// 	else if(s_axis_cmd.ready & s_axis_cmd.valid)begin
	// 		cmd_delay_en						<= cmd_delay_en + 1;
	// 	end		
	// 	else begin
	// 		cmd_delay_en						<= cmd_delay_en;
	// 	end
	// end
	


	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		cmd_delay_sum						<= 32'b0;
	// 	end 
	// 	else if(cmd_delay_en == 1)begin
	// 		cmd_delay_sum						<= cmd_delay_sum + 1'b1;
	// 	end
	// 	else begin
	// 		cmd_delay_sum						<= cmd_delay_sum;
	// 	end
	// end

	// assign status_reg[2]					= cmd_delay_sum;

	// assign status_reg[3]   					= fifo_cmd_count;
	// assign status_reg[4]   					= wr_tcp_valid_sum;
	// assign status_reg[5]   					= wr_tcp_ready_sum;



	ila_kvs_c send (
		.clk(clk), // input wire clk
	
	
		.probe0(state), // input wire [4:0]  probe0  
		.probe1(s_state), // input wire [4:0]  probe1 
		.probe2(axis_dma_read_cmd.ready), // input wire [0:0]  probe2 
		.probe3(axis_dma_read_cmd.valid), // input wire [0:0]  probe3 
		.probe4(axis_dma_read_cmd.address), // input wire [63:0]  probe4 
		.probe5(axis_dma_read_cmd.length), // input wire [31:0]  probe5 
		.probe6(axis_dma_read_data.ready), // input wire [0:0]  probe6 
		.probe7(axis_dma_read_data.valid), // input wire [0:0]  probe7 
		.probe8(axis_dma_read_data.data) // input wire [63:0]  probe8
	);


endmodule
