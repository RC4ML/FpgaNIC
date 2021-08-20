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

module dma_read_data_to_tcp#(
    parameter  MAX_SESSION_NUM  = 32 
)( 

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
	axis_meta.slave             s_axis_conn_send,   //send conn req
	axis_meta.slave				s_axis_conn_ack,    //send conn ack req 

    // axis_meta.slave    			s_axis_tcp_send_write_cnt,   //dont send to net
	axis_meta.slave    			s_axis_send_read_cnt,		//send to net
	// axis_meta.slave    			axis_tcp_recv_read_cnt,		//dont send to net
	axis_meta.slave				s_axis_cmd, 				//send
	input wire[15:0][31:0]		control_reg,
	output wire[7:0][31:0]		status_reg

	
	);
	

   


///////////////////////dma_ex debug//////////

	axi_stream								axis_tx_data();
	reg [63:0]            					dma_base_addr;       
	reg [63:0]            					session_base_addr[MAX_SESSION_NUM-1:0];                   
	reg [31:0]            					dma_session_length;
	reg [31:0]								dma_info_length;
	reg [31:0]								session_num;	                    

	reg [63:0]								read_count;

	localparam [3:0]		IDLE 			= 4'h1,
							START			= 4'h2,
							READ_CMD		= 4'h3,
							READ_DATA		= 4'h4,
							CON_SEND		= 4'h5,
							CON_SEND_DATA	= 4'h6,
							ACK_SEND		= 4'h7,
							ACK_SEND_DATA	= 4'h8,
							READ_CNT		= 4'h9,
							READ_CNT_DATA	= 4'ha,	
							SEND_CTRL		= 4'hb,
							SEND_CTRL_DATA	= 4'hc,																				
							JUDGE			= 4'hd,
							SEND_METADATA   = 4'he;	

	reg [3:0]								state;							

	reg 									rd_start_r,rd_start_rr;
	reg [31:0]								data_cnt,rd_data_cnt;
	reg [31:0]								data_cnt_minus;

    reg [31:0]                              current_addr;
	reg [31:0]                              current_length,tmp_length;
	reg [15:0]								current_session_id;
	reg [4:0]								current_buffer_id;

	reg [511:0]								tcp_tx_data;

	



	always @(posedge clk)begin
		// rd_start_r							<= control_reg[8][0];
		// rd_start_rr							<= rd_start_r;
		dma_base_addr						<= {control_reg[1],control_reg[0]};
		dma_session_length					<= control_reg[2];
		dma_info_length						<= control_reg[3];
		session_num							<= control_reg[4];
		// write_count							<= control_reg[9];
	end

	genvar i;
	generate
		for(i = 0; i < MAX_SESSION_NUM; i = i + 1) begin
			always@(posedge clk)begin
				session_base_addr[i]		<= dma_base_addr + dma_info_length + i * dma_session_length;
			end
		end
	endgenerate





	//////////////send conn cmd ///////////
	reg 									send_conn_en;	
	reg [20:0]								send_conn_data;	


	always@(posedge clk)begin
		if(~rstn)begin
			send_conn_en					<= 1'b0;
		end
		else if(s_axis_conn_send.ready & s_axis_conn_send.valid)begin
			send_conn_en					<= 1'b1;
		end
		else if(state == CON_SEND)begin
			send_conn_en					<= 1'b0;
		end		
		else begin
			send_conn_en					<= send_conn_en;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			send_conn_data					<= 1'b0;
		end
		else if(s_axis_conn_send.ready & s_axis_conn_send.valid)begin
			send_conn_data					<= s_axis_conn_send.data;
		end	
		else begin
			send_conn_data					<= send_conn_data;
		end
	end

	assign s_axis_conn_send.ready = ~send_conn_en;

////////////////////////////////////////////////////////
	//////////////ack conn cmd ///////////
	reg 									ack_conn_en;	
	reg [21:0]								ack_conn_data;	


	always@(posedge clk)begin
		if(~rstn)begin
			ack_conn_en					<= 1'b0;
		end
		else if(s_axis_conn_ack.ready & s_axis_conn_ack.valid)begin
			ack_conn_en					<= 1'b1;
		end
		else if(state == ACK_SEND)begin
			ack_conn_en					<= 1'b0;
		end		
		else begin
			ack_conn_en					<= ack_conn_en;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			ack_conn_data					<= 1'b0;
		end
		else if(s_axis_conn_ack.ready & s_axis_conn_ack.valid)begin
			ack_conn_data					<= s_axis_conn_ack.data;
		end	
		else begin
			ack_conn_data					<= ack_conn_data;
		end
	end

	assign s_axis_conn_ack.ready = ~ack_conn_en;

////////////////////////////////////////////////////////
	//////////////read cnt ///////////
	reg 									read_cnt_en;	
	reg [79:0]								read_cnt_data;	


	always@(posedge clk)begin
		if(~rstn)begin
			read_cnt_en					<= 1'b0;
		end
		else if(s_axis_send_read_cnt.ready & s_axis_send_read_cnt.valid)begin
			read_cnt_en					<= 1'b1;
		end
		else if(state == READ_CNT)begin
			read_cnt_en					<= 1'b0;
		end		
		else begin
			read_cnt_en					<= read_cnt_en;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			read_cnt_data					<= 1'b0;
		end
		else if(s_axis_send_read_cnt.ready & s_axis_send_read_cnt.valid)begin
			read_cnt_data					<= s_axis_send_read_cnt.data;
		end	
		else begin
			read_cnt_data					<= read_cnt_data;
		end
	end

	assign s_axis_send_read_cnt.ready = ~read_cnt_en;

////////////////////////////////////////////////////////	
////////////////////////////////////////////////////////	
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
		fifo_dma_cmd_wr_data					<= s_axis_cmd.data[95:0];
		fifo_dma_cmd_wr_en						<= s_axis_cmd.ready & s_axis_cmd.valid;
	end

	blockram_fifo #( 
		.FIFO_WIDTH      ( 96 ), //64 
		.FIFO_DEPTH_BITS ( 9 )  //determine the size of 16  13
	) inst_dma_cmd_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_dma_cmd_wr_en     ), //or one cycle later...
	.din        (fifo_dma_cmd_wr_data ),
	.almostfull (fifo_dma_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_dma_cmd_rd_en     ),
	.dout       (fifo_dma_cmd_rd_data   ),
	.valid      (fifo_dma_cmd_rd_valid	),
	.empty      (fifo_dma_cmd_empty     ),
	.count      (fifo_dma_cmd_count   )
	);

////////////////////////////////////////////////////////

	reg [3:0]							dma_cmd_state;
	reg [31:0]							dma_cmd_addr,dma_cmd_length;
	reg [4:0]							dma_cmd_buffer_id;

	always @(posedge clk)begin
		if(~rstn)begin
			dma_cmd_state				<= IDLE;
		end
		else begin
			fifo_dma_cmd_rd_en			<= 1'b0;
			case(dma_cmd_state)				
				IDLE:begin
					if(~fifo_dma_cmd_empty)begin
						fifo_dma_cmd_rd_en	<= 1'b1;
						dma_cmd_state			<= START;
					end
					else begin
						dma_cmd_state			<= IDLE;
					end
				end
				START:begin
					if(fifo_dma_cmd_rd_valid)begin
						dma_cmd_state       <= READ_CMD;
						dma_cmd_addr		<= fifo_dma_cmd_rd_data[63:32];
						dma_cmd_length		<= fifo_dma_cmd_rd_data[31:0];
						dma_cmd_buffer_id	<= fifo_dma_cmd_rd_data[84:80];
					end
					else begin
						dma_cmd_state			<= START;
					end
				end			
				READ_CMD:begin
					if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
						dma_cmd_state			<= IDLE;
					end
					else begin
						dma_cmd_state			<= READ_CMD;
					end
				end
			endcase
		end
	end

////////////////////////////////////////////////////////	
	//////////////dma data cmd buffer ///////////
	reg 									fifo_dma_data_wr_en;
	reg 									fifo_dma_data_rd_en;			
	wire 									fifo_dma_data_almostfull;	
	wire 									fifo_dma_data_empty;
	reg [31:0]								fifo_dma_data_wr_data;	
	wire [31:0]								fifo_dma_data_rd_data;
	wire 									fifo_dma_data_rd_valid;
	wire [9:0]								fifo_dma_data_count;	

	always@(posedge clk)begin
		fifo_dma_data_wr_data					<= s_axis_cmd.data[31:0];
		fifo_dma_data_wr_en						<= s_axis_cmd.ready & s_axis_cmd.valid;
	end

	blockram_fifo #( 
		.FIFO_WIDTH      ( 32 ), //64 
		.FIFO_DEPTH_BITS ( 9 )  //determine the size of 16  13
	) inst_dma_data_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_dma_data_wr_en     ), //or one cycle later...
	.din        (fifo_dma_data_wr_data ),
	.almostfull (fifo_dma_data_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_dma_data_rd_en     ),
	.dout       (fifo_dma_data_rd_data   ),
	.valid      (fifo_dma_data_rd_valid	),
	.empty      (fifo_dma_data_empty     ),
	.count      (fifo_dma_data_count   )
	);

////////////////////////////////////////////////////////
	reg [3:0]							dma_data_state;
	reg [31:0]							dma_data_addr,dma_data_length;
	reg [4:0]							dma_data_buffer_id;

	always @(posedge clk)begin
		if(~rstn)begin
			dma_data_state				<= IDLE;
		end
		else begin
			fifo_dma_data_rd_en			<= 1'b0;
			case(dma_data_state)				
				IDLE:begin
					if(~fifo_dma_data_empty)begin
						fifo_dma_data_rd_en	<= 1'b1;
						dma_data_state			<= START;
					end
					else begin
						dma_data_state			<= IDLE;
					end
				end
				START:begin
					if(fifo_dma_data_rd_valid)begin
						dma_data_state       <= READ_CMD;
						dma_data_length		<= fifo_dma_data_rd_data;
					end
					else begin
						dma_data_state			<= START;
					end
				end			
				READ_CMD:begin
					if(axis_dma_data.last)begin
						dma_data_state			<= IDLE;
					end
					else begin
						dma_data_state			<= READ_CMD;
					end
				end
			endcase
		end
	end


    always @(posedge clk)begin
		data_cnt_minus						<= (dma_data_length>>>6)-1;
	end




	//////////////cmd buffer ///////////
	wire 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_almostfull;	
	wire 									fifo_cmd_empty;	
	wire [95:0]								fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;
	wire [9:0]								fifo_cmd_count;	

	assign fifo_cmd_wr_en					= s_axis_cmd.ready & s_axis_cmd.valid;

	assign s_axis_cmd.ready 				= ~fifo_cmd_almostfull;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 96 ), //64 
		.FIFO_DEPTH_BITS ( 9 )  //determine the size of 16  13
	) inst_cmd_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_cmd_wr_en     ), //or one cycle later...
	.din        (s_axis_cmd.data[95:0] ),
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
	reg										read_cmd_enable;
	reg[31:0]								delay_cycle;  // cycle = length *25/1024   th = 10.24G 
	reg[31:0]								delay_big_cycle; //26:9.85GB/s  25:10.24GB/s    24:10.67GB/s   23:11.13GB/s   
	reg[31:0]								delay_cnt;
	reg[31:0]								delay_big_cnt;


	always@(posedge clk)begin
		if(~rstn)begin
			delay_cycle						<= 1'b0;
		end
		else if(fifo_cmd_rd_valid)begin
			delay_cycle						<= fifo_cmd_rd_data[31:10];
		end
		else begin
			delay_cycle						<= delay_cycle;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			delay_cnt						<= 1'b1;		
		end
		else if(delay_cnt == delay_cycle)begin
			delay_cnt						<= 1'b0;
		end
		else if(~read_cmd_enable)begin
			delay_cnt						<= delay_cnt + 1'b1;
		end
		else begin
			delay_cnt						<= delay_cnt;
		end
	end

	always@(posedge clk)begin
		delay_big_cycle						<= control_reg[6];
	end

	always@(posedge clk)begin
		if(~rstn)begin
			delay_big_cnt					<= 1'b0;		
		end
		else if(delay_big_cnt == delay_big_cycle)begin
			delay_big_cnt					<= 1'b0;
		end
		else if((delay_cnt == delay_cycle) && ~read_cmd_enable)begin
			delay_big_cnt					<= delay_big_cnt + 1'b1;
		end
		else begin
			delay_big_cnt					<= delay_big_cnt;
		end
	end


	always@(posedge clk)begin
		if(~rstn)begin
			read_cmd_enable					<= 1'b1;		
		end
		else if(fifo_cmd_rd_valid)begin
			read_cmd_enable					<= 1'b0;
		end
		else if(delay_big_cnt == delay_big_cycle)begin
			read_cmd_enable					<= 1'b1;
		end
		else begin
			read_cmd_enable					<= read_cmd_enable;
		end
	end

/////////////////////////////////////////////////////////////



	reg 									tx_metadata_valid;	

	assign	axis_dma_read_cmd.address	    = session_base_addr[dma_cmd_buffer_id] + dma_cmd_addr;
	assign	axis_dma_read_cmd.length	    = dma_cmd_length; 
	assign 	axis_dma_read_cmd.valid			= (dma_cmd_state == READ_CMD); 
	
	assign 	m_axis_tx_metadata.data			= {current_length,current_session_id};
	assign 	m_axis_tx_metadata.valid		= (state == CON_SEND) || (state == ACK_SEND) || (state == READ_CNT) || (state == SEND_CTRL) || (state == SEND_METADATA);

	assign s_axis_tx_status.ready 			= 1;

	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		tx_metadata_valid				<= 1'b0;
	// 	end
	// 	else if((state == CON_SEND) || (state == ACK_SEND) || (state == READ_CNT) || (state == SEND_CTRL))begin
	// 		tx_metadata_valid				<= 1'b1;
	// 	end
	// 	else if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
	// 		tx_metadata_valid				<= 1'b1;
	// 	end
	// 	else if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
	// 		tx_metadata_valid				<= 1'b0;
	// 	end		
	// 	else begin
	// 		tx_metadata_valid				<= tx_metadata_valid;
	// 	end
	// end

	always@(posedge clk)begin
		if(~rstn)begin
			tcp_tx_data						<= 1'b0;
		end
		else if(state == CON_SEND)begin
			tcp_tx_data						<= 512'b0;
		end
		else if(state == ACK_SEND)begin
			tcp_tx_data						<= {495'h0,ack_conn_data[21],16'h1};
		end
		else if(state == READ_CNT)begin
			tcp_tx_data						<= {432'h0,read_count,16'h2};
		end		
		else if(state == SEND_CTRL)begin
			tcp_tx_data						<= {432'h0,tmp_length,16'h3};
		end		
		else begin
			tcp_tx_data						<= tcp_tx_data;
		end
	end
	


	axi_stream 								axis_dma_data();

	assign axis_dma_data.valid 		= axis_dma_read_data.valid && (dma_data_state == READ_CMD);
	assign axis_dma_read_data.ready = axis_dma_data.ready && (dma_data_state == READ_CMD);
	assign axis_dma_data.data 		= axis_dma_read_data.data;
	assign axis_dma_data.last 		= (data_cnt == data_cnt_minus) && axis_dma_read_data.ready && axis_dma_read_data.valid;
	assign axis_dma_data.keep 		= axis_dma_read_data.keep;


	assign m_axis_tx_data.valid		= (axis_tx_data.valid && ((state == READ_DATA) || (state == JUDGE))) || (state == CON_SEND_DATA) || (state == ACK_SEND_DATA) || (state == READ_CNT_DATA) || (state == SEND_CTRL_DATA);
	assign m_axis_tx_data.keep		= ((state == CON_SEND_DATA) || (state == ACK_SEND_DATA) || (state == READ_CNT_DATA) || (state == SEND_CTRL_DATA)) ? 64'hffff_ffff_ffff_ffff : axis_tx_data.keep ;
	assign m_axis_tx_data.last		= ((state == CON_SEND_DATA) || (state == ACK_SEND_DATA) || (state == READ_CNT_DATA) || (state == SEND_CTRL_DATA)) ? (m_axis_tx_data.ready & m_axis_tx_data.valid) : axis_tx_data.last;
	assign m_axis_tx_data.data		= ((state == READ_DATA) || (state == JUDGE)) ? (axis_tx_data.data) : tcp_tx_data;
	assign axis_tx_data.ready 		= m_axis_tx_data.ready && ((state == READ_DATA) || (state == JUDGE));



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

	// always @(posedge clk)begin
	// 	if(~rstn)begin
	// 		read_count 						<= 1'b0;
	// 	end
	// 	else if( (state == END))begin
	// 		read_count						<= read_count + 1'b1;
	// 	end
	// 	else begin
	// 		read_count						<= read_count;
	// 	end		
	// end
	
	// assign status_reg[0] = read_count;


	always @(posedge clk)begin
		if(~rstn)begin
			state						<= IDLE;
		end
		else begin
			fifo_cmd_rd_en				<= 1'b0;
			case(state)				
				IDLE:begin
					if(send_conn_en)begin
						current_length		<= 32'h40;
						current_session_id	<= send_conn_data[15:0];
						state				<= CON_SEND;
					end
					else if(ack_conn_en)begin
						current_length		<= 32'h40;
						current_session_id	<= ack_conn_data[15:0];
						state				<= ACK_SEND;						
					end
					else if(read_cnt_en)begin
						current_length		<= 32'h40;
						read_count			<= read_cnt_data[63:0];
						current_session_id	<= read_cnt_data[79:64];
						state				<= READ_CNT;						
					end
					else if(~fifo_cmd_empty && read_cmd_enable)begin
						fifo_cmd_rd_en	<= 1'b1;
						state			<= START;
					end
					else begin
						state			<= IDLE;
					end
				end
				CON_SEND:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						state				<= CON_SEND_DATA;
					end
					else begin
						state				<= CON_SEND;
					end				
				end
				CON_SEND_DATA:begin
					if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
						state				<= IDLE;
					end						
					else begin
						state				<= CON_SEND_DATA;
					end
				end
				ACK_SEND:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						state				<= ACK_SEND_DATA;
					end
					else begin
						state				<= ACK_SEND;
					end				
				end
				ACK_SEND_DATA:begin
					if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
						state				<= IDLE;
					end						
					else begin
						state				<= ACK_SEND_DATA;
					end
				end
				READ_CNT:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						state				<= READ_CNT_DATA;
					end
					else begin
						state				<= READ_CNT;
					end				
				end
				READ_CNT_DATA:begin
					if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
						state				<= IDLE;
					end						
					else begin
						state				<= READ_CNT_DATA;
					end
				end
				START:begin
					if(fifo_cmd_rd_valid)begin
						state           	<= SEND_CTRL;
						current_addr		<= fifo_cmd_rd_data[63:32];
						tmp_length			<= fifo_cmd_rd_data[31:0];
						current_length		<= 32'h40;
						current_session_id	<= fifo_cmd_rd_data[79:64];
						current_buffer_id	<= fifo_cmd_rd_data[84:80];
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
						state				<= SEND_METADATA;
						current_length		<= tmp_length;
					end						
					else begin
						state				<= SEND_CTRL_DATA;
					end
				end				
				// READ_CMD:begin
				// 	if(axis_dma_read_cmd.ready & axis_dma_read_cmd.valid)begin
				// 		state			<= SEND_METADATA;
				// 	end
				// 	else begin
				// 		state			<= READ_CMD;
				// 	end
				// end
				SEND_METADATA:begin
					if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
						state				<= JUDGE;
					end
					else begin
						state				<= SEND_METADATA;
					end				
				end
				// READ_DATA:begin
				// 	if(axis_dma_data.last)begin
				// 		state			<= JUDGE;
				// 	end	
				// 	else begin
				// 		state			<= READ_DATA;
				// 	end
                // end
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
		else if(wr_tcp_data_cnt == wr_length_minus)begin
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
		else if((m_axis_tx_data.ready & m_axis_tx_data.valid) && ((state == READ_DATA) || (state == JUDGE)))begin
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

	always@(posedge clk)begin
		if(~rstn)begin
			wr_tcp_valid_sum				<= 32'b0;
		end 
		else if(wr_tcp_en && axis_dma_read_data.valid && (~axis_dma_read_data.ready))begin
			wr_tcp_valid_sum				<= wr_tcp_valid_sum + 1'b1;
		end
		else begin
			wr_tcp_valid_sum				<= wr_tcp_valid_sum;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_tcp_ready_sum				<= 32'b0;
		end 
		else if(wr_tcp_en && (~axis_dma_read_data.valid) && axis_dma_read_data.ready)begin
			wr_tcp_ready_sum				<= wr_tcp_ready_sum + 1'b1;
		end
		else begin
			wr_tcp_ready_sum				<= wr_tcp_ready_sum;
		end
	end

	assign status_reg[1]					= wr_tcp_sum;

	////////////////////////////////////////////////////////////////
	reg [1:0]								cmd_delay_en;
	reg [31:0]								cmd_delay_sum;	

	always@(posedge clk)begin
		if(~rstn)begin
			cmd_delay_en						<= 1'b0;
		end  
		else if(cmd_delay_en == 2)begin
			cmd_delay_en						<= cmd_delay_en;
		end
		else if(s_axis_cmd.ready & s_axis_cmd.valid)begin
			cmd_delay_en						<= cmd_delay_en + 1;
		end		
		else begin
			cmd_delay_en						<= cmd_delay_en;
		end
	end
	


	always@(posedge clk)begin
		if(~rstn)begin
			cmd_delay_sum						<= 32'b0;
		end 
		else if(cmd_delay_en == 1)begin
			cmd_delay_sum						<= cmd_delay_sum + 1'b1;
		end
		else begin
			cmd_delay_sum						<= cmd_delay_sum;
		end
	end

	assign status_reg[2]					= cmd_delay_sum;

	assign status_reg[3]   					= fifo_cmd_count;
	assign status_reg[4]   					= wr_tcp_valid_sum;
	assign status_reg[5]   					= wr_tcp_ready_sum;


	ila_0 rx (
		.clk(clk), // input wire clk
	
	
		.probe0(axis_dma_read_cmd.valid), // input wire [0:0]  probe0  
		.probe1(axis_dma_read_cmd.ready), // input wire [0:0]  probe1 
		.probe2(axis_dma_read_cmd.address), // input wire [63:0]  probe2 
		.probe3(axis_dma_read_cmd.length), // input wire [31:0]  probe3 
		.probe4(axis_dma_data.valid), // input wire [0:0]  probe4 
		.probe5(axis_dma_data.ready), // input wire [0:0]  probe5 
		.probe6(axis_dma_data.last), // input wire [0:0]  probe6 
		.probe7(axis_dma_data.data[31:0]), // input wire [511:0]  probe7
		.probe8(fifo_dma_cmd_wr_en),
		.probe9(state), // input wire [31:0]  probe8 
		.probe10(fifo_dma_cmd_rd_en), // input wire [0:0]  probe10 
	   .probe11(fifo_dma_cmd_rd_valid), // input wire [0:0]  probe11 
	   .probe12(dma_cmd_addr), // input wire [87:0]  probe12
		.probe13(dma_cmd_state ), // input wire [31:0]  probe9	
		.probe14(s_axis_cmd.data[95:0]), // input wire [31:0]  probe9
		.probe15(wr_tcp_ready_sum) // input wire [31:0]  probe9	
	);

	// ila_0 tx (
	// 	.clk(clk), // input wire clk
	
	
	// 	.probe0(axis_dma_write_cmd.valid), // input wire [0:0]  probe0  
	// 	.probe1(axis_dma_write_cmd.ready), // input wire [0:0]  probe1 
	// 	.probe2(axis_dma_write_cmd.address), // input wire [63:0]  probe2 
	// 	.probe3(axis_dma_write_cmd.length), // input wire [31:0]  probe3 
	// 	.probe4(axis_dma_write_data.valid), // input wire [0:0]  probe4 
	// 	.probe5(axis_dma_write_data.ready), // input wire [0:0]  probe5 
	// 	.probe6(axis_dma_write_data.last), // input wire [0:0]  probe6 
	// 	.probe7(axis_dma_write_data.data), // input wire [511:0]  probe7
	// 	.probe8(wr_th_sum), // input wire [31:0]  probe8 
	// 	.probe9({state,gpu_write_cnt[27:0]}) // input wire [31:0]  probe9		
	// );








endmodule
