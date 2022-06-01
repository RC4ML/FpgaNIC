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

module dma_write_data_from_tcp#(
	parameter  MAX_SESSION_NUM  = 8,
	parameter  DDR_SESSION_LENGTH = 1024*1024*64 
)( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,

    //DMA Commands
    axis_mem_cmd.master         axis_dma_write_cmd,

    //DMA Data streams      
    axi_stream.master           axis_dma_write_data,

	//tcp send
    // axis_meta.slave    			s_axis_notifications,
    // axis_meta.master     		m_axis_read_package,
    
    axis_meta.slave    			s_axis_rx_metadata,
    axi_stream.slave   			s_axis_rx_data,

	//control cmd
	axis_meta.slave				s_axis_set_buffer_id,
	axis_meta.master			m_axis_conn_ack_recv,
	//control reg
	input wire[15:0][31:0]		control_reg,
	output wire[7:0][31:0]		status_reg

    );
    

///////////////////////dma_ex debug//////////

	localparam [4:0]		IDLE 				= 5'h0,
							START				= 5'h1,
							JUDGE				= 5'h2,
							JUDGE1				= 5'h3,	
							READ_CTRL			= 5'h4,
							INFO_WRITE			= 5'h5,
							INFO_WRITE_DATA		= 5'h6,
							ACK_RECV			= 5'h7,
							WRITE_INFO_CMD		= 5'h8,
							WRITE_INFO_DATA		= 5'h9,
							WRITE_END_DATA		= 5'ha,

							TCP_DATA_RECV		= 5'hc,
							WRITE_CMD			= 5'hd,							
							WRITE_CTRL_DATA		= 5'he,
							WRITE_DATA			= 5'hf,
							END         		= 5'h10;	
				






	reg [4:0]								state,w_state;							

	reg [63:0]            					dma_base_addr;    
	reg [63:0]								dma_recv_info_addr;
	reg [63:0]            					session_base_addr[MAX_SESSION_NUM-1:0];                   
	reg [31:0]            					dma_session_length;
	reg [31:0]								dma_info_length;
	reg [63:0] 								dma_session_almost_addr[MAX_SESSION_NUM-1:0];
	reg [63:0]								session_addr[MAX_SESSION_NUM-1:0];
	reg [31:0]								dma_session_max_package;
	reg [31:0]								session_num;
	reg [31:0]								package_length;		
	reg [31:0]								remain_length[MAX_SESSION_NUM-1:0];
	reg [31:0]								dma_remain_length[MAX_SESSION_NUM-1:0];
	reg [16:0]								session_id[MAX_SESSION_NUM-1:0];

	// reg [31:0]								ddr_base_addr[MAX_SESSION_NUM-1:0];						
	// reg [31:0]								ddr_write_addr[MAX_SESSION_NUM-1:0];
	// reg [31:0]								ddr_read_addr[MAX_SESSION_NUM-1:0];	

	reg 									wr_start_r,wr_start_rr;
	reg [31:0]								data_cnt;

    reg [63:0]                              current_addr;
	reg [15:0]                              current_length,data_minus;
	reg [15:0]								current_session_id;
	reg [31:0]								des_ip_addr;
	reg [15:0]								des_port;
	reg [7:0]								session_close_flag;
	reg [5:0]								current_buffer_id;
	reg [31:0]								info_count;
	reg [255:0]								info_data;
	reg [79:0]								ctrl_data;

	reg[31:0]								current_data_count;
	wire[31:0]								dma_info_count,dma_data_count[MAX_SESSION_NUM-1:0];	

	reg 									dma_write_cmd_valid;
	reg[31:0]								current_dma_length;	
	reg[4:0]								dma_buffer_id,write_to_dma_enable;
				

//////////////////////////////////set buffer id////////////////
	assign s_axis_set_buffer_id.ready				= 1'b1;

	//////////////notifications buffer ///////////
	wire 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_empty;	
	wire 									fifo_cmd_almostfull;		
	wire [87:0]								fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;			

	assign s_axis_rx_metadata.ready 		= ~fifo_cmd_almostfull;
	// assign s_axis_rx_metadata.ready			=  1'b1;
	assign fifo_cmd_wr_en					= s_axis_rx_metadata.ready && s_axis_rx_metadata.valid;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 88 ), //64 
		.FIFO_DEPTH_BITS ( 10 )  //determine the size of 16  13
	) inst_tcp_notice_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_cmd_wr_en     ), //or one cycle later...
	.din        (s_axis_rx_metadata.data ),
	.almostfull (fifo_cmd_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_cmd_rd_en     ),
	.dout       (fifo_cmd_rd_data   ),
	.valid      (fifo_cmd_rd_valid	),
	.empty      (fifo_cmd_empty     ),
	.count      (   )
	);

////////////////////////////////////////////////////////
//////////////dma information buffer ///////////

	axi_stream 								axis_tcp_info();
	axi_stream								axis_tcp_info_to_dma();
	wire 									tcp_info_to_dma_last;		

	assign axis_tcp_info.keep				= 64'hffff_ffff_ffff_ffff;
	assign axis_tcp_info.last				= 1'b0;
	assign axis_tcp_info.data				= {256'b0,info_data};
	assign axis_tcp_info.valid				= state == INFO_WRITE_DATA;

	assign axis_tcp_info_to_dma.ready 		= w_state == WRITE_INFO_DATA;


	axis_data_fifo_512_d1024 inst_dma_info_fifo (
		.s_axis_aresetn(rstn),          // input wire s_axis_aresetn
		.s_axis_aclk(clk),                // input wire s_axis_aclk
		.s_axis_tvalid(axis_tcp_info.valid),            // input wire s_axis_tvalid
		.s_axis_tready(axis_tcp_info.ready),            // output wire s_axis_tready
		.s_axis_tdata(axis_tcp_info.data),              // input wire [511 : 0] s_axis_tdata
		.s_axis_tkeep(axis_tcp_info.keep),              // input wire [63 : 0] s_axis_tkeep
		.s_axis_tlast(axis_tcp_info.last),              // input wire s_axis_tlast
		.m_axis_tvalid(axis_tcp_info_to_dma.valid),            // output wire m_axis_tvalid
		.m_axis_tready(axis_tcp_info_to_dma.ready),            // input wire m_axis_tready
		.m_axis_tdata(axis_tcp_info_to_dma.data),              // output wire [511 : 0] m_axis_tdata
		.m_axis_tkeep(axis_tcp_info_to_dma.keep),              // output wire [63 : 0] m_axis_tkeep
		.m_axis_tlast(),              // output wire m_axis_tlast
		.axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
		.axis_rd_data_count(dma_info_count)  // output wire [31 : 0] axis_rd_data_count
	  );





////////////////////////////////////////////////////////	


	always @(posedge clk)begin
		if(~rstn)begin
			dma_recv_info_addr 			        <= 1'b0;
		end
		else if(wr_start_r & ~wr_start_rr)begin
			dma_recv_info_addr					<= dma_base_addr;
		end
        else if(dma_recv_info_addr >= (dma_base_addr + dma_info_length))begin
            dma_recv_info_addr                    <= dma_base_addr;
        end
		else if((w_state == WRITE_INFO_CMD) && axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
			dma_recv_info_addr					<= dma_recv_info_addr + 32'h40;
		end
		else begin
			dma_recv_info_addr					<= dma_recv_info_addr;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
			info_count 			        		<= 1'b1;
		end
        else if(info_count == 32'hffff_ffff)begin
            info_count                    		<= 1'b1;
        end
		else if((state == INFO_WRITE_DATA) && axis_tcp_info.ready & axis_tcp_info.valid)begin
			info_count							<= info_count + 1'b1;
		end
		else begin
			info_count							<= info_count;
		end		
	end	

	always @(posedge clk)begin
		if(~rstn)begin
			info_data 			        		<= 1'b0;
		end
		else if(state == INFO_WRITE)begin
			if(ctrl_data[2:0] == 0)begin
				info_data						<= {64'h0,16'h0,des_port,des_ip_addr,16'h0,current_session_id,26'h0,current_buffer_id,32'h0,info_count};
			end
			else if(ctrl_data[2:0] == 2)begin
				info_data						<= {ctrl_data[79:16],16'h0,des_port,des_ip_addr,16'h0,current_session_id,26'h0,current_buffer_id,32'h2,info_count};
			end
			else if(ctrl_data[2:0] == 4)begin
				info_data						<= {64'h0,16'h0,des_port,des_ip_addr,16'h0,current_session_id,26'h0,current_buffer_id,32'h1,info_count};
			end
			else begin
				info_data						<= info_data;
			end
		end
		else begin
			info_data							<= info_data;
		end
	end		
////////////////////////////////////////////////////////read tcp data
	// reg 									read_package_valid;	
	// reg 									read_ctrl_r;


	
	// assign 	m_axis_read_package.data		= {current_length,current_session_id};
	// assign 	m_axis_read_package.valid		= read_package_valid;

	// always@(posedge clk)begin
	// 	read_ctrl_r							<= state == READ_CTRL;
	// end

	// always@(posedge clk)begin
	// 	if(~rstn)begin
	// 		read_package_valid				<= 1'b0;
	// 	end
	// 	else if((state == READ_CTRL) && (~read_ctrl_r))begin
	// 		read_package_valid				<= 1'b1;
	// 	end
	// 	else if(state == WRITE_CMD)begin
	// 		read_package_valid				<= 1'b1;
	// 	end
	// 	else if(m_axis_read_package.ready & m_axis_read_package.valid)begin
	// 		read_package_valid				<= 1'b0;
	// 	end		
	// 	else begin
	// 		read_package_valid				<= read_package_valid;
	// 	end
	// end


/////////////////////// //write data to dma



	always @(posedge clk)begin
		if(~rstn)begin
			wr_start_r 			        	<= 1'b0;
		end
		else if(fifo_cmd_wr_en)begin
			wr_start_r						<= 1'b1;
        end
		else begin
			wr_start_r						<= wr_start_r;
		end		
	end

	// axi_stream 								axis_dma_data[MAX_SESSION_NUM]();
	// axi_stream								axis_rx_data[MAX_SESSION_NUM]();
	wire 									axis_dma_data_ready[MAX_SESSION_NUM-1:0];
	wire 									axis_dma_data_valid[MAX_SESSION_NUM-1:0];
	wire 									axis_dma_data_last[MAX_SESSION_NUM-1:0];
	wire [511:0]							axis_dma_data_data[MAX_SESSION_NUM-1:0];
	wire [63:0]								axis_dma_data_keep[MAX_SESSION_NUM-1:0];	

	wire 									axis_rx_data_ready[MAX_SESSION_NUM-1:0];
	wire 									axis_rx_data_valid[MAX_SESSION_NUM-1:0];
	wire 									axis_rx_data_last[MAX_SESSION_NUM-1:0];
	wire [511:0]							axis_rx_data_data[MAX_SESSION_NUM-1:0];
	wire [63:0]								axis_rx_data_keep[MAX_SESSION_NUM-1:0];


	assign s_axis_rx_data.ready 			= (state == READ_CTRL) ? 1 : ((state == WRITE_DATA) ? axis_rx_data_ready[current_buffer_id] : 0);

	reg [255:0]								data_info;
	reg [31:0]								packet_counter;

	always@(posedge clk)begin
		if(~rstn)begin
			packet_counter					<= 0;
		end
		else if((w_state == WRITE_CMD) && axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
			packet_counter					<= packet_counter + 1'b1;
		end
		else begin
			packet_counter					<= packet_counter;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			data_info						<= 0;
		end
		else if(w_state == WRITE_CMD)begin
			data_info						<= {64'b0,packet_counter,27'b0,write_to_dma_enable,16'b0,des_port,des_ip_addr,16'b0,current_dma_length,16'b0,current_session_id};
		end
		else begin
			data_info						<= data_info;
		end
	end



	assign	axis_dma_write_cmd.address	    = current_addr;
	assign	axis_dma_write_cmd.length	    = current_dma_length; 
	assign 	axis_dma_write_cmd.valid		= dma_write_cmd_valid; 

	assign axis_dma_write_data.valid	= (axis_dma_data_valid[dma_buffer_id] && (w_state == WRITE_DATA)) || (w_state == WRITE_END_DATA) || (w_state == WRITE_CTRL_DATA) || ((w_state == WRITE_INFO_DATA) && axis_tcp_info_to_dma.valid);
	assign axis_dma_write_data.keep		= (w_state == WRITE_DATA) ?  axis_dma_data_keep[dma_buffer_id] : 64'hffff_ffff_ffff_ffff;
	assign axis_dma_write_data.last		= ((w_state == WRITE_END_DATA) || (w_state == WRITE_INFO_DATA)) && (axis_dma_write_data.ready & axis_dma_write_data.valid);
	assign axis_dma_write_data.data		= (w_state == WRITE_DATA) ? (axis_dma_data_data[dma_buffer_id]) : ((w_state == WRITE_INFO_DATA) ? axis_tcp_info_to_dma.data : data_info);




	genvar i;
	generate
		for(i = 0; i < MAX_SESSION_NUM; i = i + 1) begin
			always@(posedge clk)begin
				session_base_addr[i]		<= dma_base_addr + dma_info_length + i * dma_session_length;
				dma_session_almost_addr[i]	<= dma_base_addr + dma_info_length + (i+1) * dma_session_length - dma_session_max_package;
			end
			always @(posedge clk)begin
				if(~rstn)begin
					session_addr[i] 			    <= 1'b0;
				end
				else if(wr_start_r & ~wr_start_rr)begin
					session_addr[i]					<= session_base_addr[i];
				end
				else if(session_addr[i] > dma_session_almost_addr[i])begin
					session_addr[i]                 <= session_base_addr[i];
				end
				else if((dma_buffer_id == i) && (w_state == WRITE_CMD) && axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
					session_addr[i]					<= session_addr[i] + current_dma_length;
				end
				else begin
					session_addr[i]					<= session_addr[i];
				end		
			end			

			always @(posedge clk)begin
				if(~rstn)begin
					remain_length[i] 			    <= 1'b0;
				end
				else if((current_buffer_id == i) && (state == TCP_DATA_RECV))begin
					remain_length[i]                <= ctrl_data[47:16];
				end
				else if((current_buffer_id == i) && (state == WRITE_CMD))begin
					remain_length[i]				<= remain_length[i] - current_length;
				end
				else begin
					remain_length[i]				<= remain_length[i];
				end		
			end

			always @(posedge clk)begin
				if(~rstn)begin
					dma_remain_length[i] 			    <= 1'b0;
				end
				else if((current_buffer_id == i) && (state == TCP_DATA_RECV) && (w_state == WRITE_CTRL_DATA) && axis_dma_write_data.ready & axis_dma_write_data.valid)begin
					dma_remain_length[i]                <= dma_remain_length[i] + ctrl_data[47:16] - current_dma_length + 16'h80;
				end				
				else if((current_buffer_id == i) && (state == TCP_DATA_RECV))begin
					dma_remain_length[i]                <= dma_remain_length[i] + ctrl_data[47:16];
				end
				else if((dma_buffer_id == i) && (w_state == WRITE_CTRL_DATA) && axis_dma_write_data.ready & axis_dma_write_data.valid)begin
					dma_remain_length[i]				<= dma_remain_length[i] - current_dma_length + 16'h80;
				end
				else begin
					dma_remain_length[i]				<= dma_remain_length[i];
				end		
			end						


			always @(posedge clk)begin
				if(~rstn)begin
					session_id[i] 			    <= 17'h1_0000;
				end
				else if((state == INFO_WRITE) && (ctrl_data[2:0] == 4) && (current_buffer_id == i))begin
					session_id[i]				<= 17'h1_0000;
				end				
				else if((s_axis_set_buffer_id.data[20:16] == i) && s_axis_set_buffer_id.valid && s_axis_set_buffer_id.ready)begin
					session_id[i]                <= {1'b0,s_axis_set_buffer_id.data[15:0]};
				end
				else begin
					session_id[i]				<= session_id[i];
				end		
			end

			assign axis_rx_data_valid[i] 		= (current_buffer_id == i) ? ((state == WRITE_DATA) ? s_axis_rx_data.valid : 0) : 0;
			assign axis_rx_data_data[i] 		= s_axis_rx_data.data;
			assign axis_rx_data_keep[i]			= s_axis_rx_data.keep;
			assign axis_rx_data_last[i]			= 0;

			assign axis_dma_data_ready[i] 		= (dma_buffer_id == i) ? (axis_dma_write_data.ready && (w_state == WRITE_DATA)) : 0;

			axis_data_fifo_512_d4096 write_data_slice_fifo (
				.s_axis_aresetn(rstn),          // input wire s_axis_aresetn
				.s_axis_aclk(clk),                // input wire s_axis_aclk
				.s_axis_tvalid(axis_rx_data_valid[i]),            // input wire s_axis_tvalid
				.s_axis_tready(axis_rx_data_ready[i]),            // output wire s_axis_tready
				.s_axis_tdata(axis_rx_data_data[i]),              // input wire [511 : 0] s_axis_tdata
				.s_axis_tkeep(axis_rx_data_keep[i]),              // input wire [63 : 0] s_axis_tkeep
				.s_axis_tlast(axis_rx_data_last[i]),              // input wire s_axis_tlast
				.m_axis_tvalid(axis_dma_data_valid[i]),            // output wire m_axis_tvalid
				.m_axis_tready(axis_dma_data_ready[i]),            // input wire m_axis_tready
				.m_axis_tdata(axis_dma_data_data[i]),              // output wire [511 : 0] m_axis_tdata
				.m_axis_tkeep(axis_dma_data_keep[i]),              // output wire [63 : 0] m_axis_tkeep
				.m_axis_tlast(axis_dma_data_last[i]),              // output wire m_axis_tlast
				.axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
				.axis_rd_data_count(dma_data_count[i])  // output wire [31 : 0] axis_rd_data_count
			  );



		end
	endgenerate


	always @(posedge clk)begin	
		wr_start_rr							<= wr_start_r;
		dma_base_addr						<= {control_reg[1],control_reg[0]};
		dma_session_length					<= control_reg[2];
		dma_info_length						<= control_reg[3];
		dma_session_max_package				<= control_reg[4];
		session_num							<= control_reg[5];
		package_length						<= control_reg[6];
	end





//////data length/////////////////////////


	always@(posedge clk)begin
		if(~rstn)begin
			write_to_dma_enable				<= 5'b10000;
		end
		else if(((dma_remain_length[0] > 0) && (dma_remain_length[0] == (dma_data_count[0] <<< 6))) || (dma_data_count[0] > package_length))begin
			write_to_dma_enable				<= 0;
		end
		else if(((dma_remain_length[1] > 0) && (dma_remain_length[1] == (dma_data_count[1] <<< 6))) || (dma_data_count[1] > package_length))begin
			write_to_dma_enable				<= 1;
		end
		else if(((dma_remain_length[2] > 0) && (dma_remain_length[2] == (dma_data_count[2] <<< 6))) || (dma_data_count[2] > package_length))begin
			write_to_dma_enable				<= 2;
		end
		else if(((dma_remain_length[3] > 0) && (dma_remain_length[3] == (dma_data_count[3] <<< 6))) || (dma_data_count[3] > package_length))begin
			write_to_dma_enable				<= 3;
		end
		else if(((dma_remain_length[4] > 0) && (dma_remain_length[4] == (dma_data_count[4] <<< 6))) || (dma_data_count[4] > package_length))begin
			write_to_dma_enable				<= 4;
		end		
		else if(((dma_remain_length[5] > 0) && (dma_remain_length[5] == (dma_data_count[5] <<< 6))) || (dma_data_count[5] > package_length))begin
			write_to_dma_enable				<= 5;
		end
		else if(((dma_remain_length[6] > 0) && (dma_remain_length[6] == (dma_data_count[6] <<< 6))) || (dma_data_count[6] > package_length))begin
			write_to_dma_enable				<= 6;
		end
		else if(((dma_remain_length[7] > 0) && (dma_remain_length[7] == (dma_data_count[7] <<< 6))) || (dma_data_count[7] > package_length))begin
			write_to_dma_enable				<= 7;
		end
		// else if(((dma_remain_length[8] > 0) && (dma_remain_length[8] == (dma_data_count[8] <<< 6))) || (dma_data_count[8] > package_length))begin
		// 	write_to_dma_enable				<= 8;
		// end
		// else if(((dma_remain_length[9] > 0) && (dma_remain_length[9] == (dma_data_count[9] <<< 6))) || (dma_data_count[9] > package_length))begin
		// 	write_to_dma_enable				<= 9;
		// end
		// else if(((dma_remain_length[10] > 0) && (dma_remain_length[10] == (dma_data_count[10] <<< 6))) || (dma_data_count[10] > package_length))begin
		// 	write_to_dma_enable				<= 10;
		// end
		// else if(((dma_remain_length[11] > 0) && (dma_remain_length[11] == (dma_data_count[11] <<< 6))) || (dma_data_count[11] > package_length))begin
		// 	write_to_dma_enable				<= 11;
		// end
		// else if(((dma_remain_length[12] > 0) && (dma_remain_length[12] == (dma_data_count[12] <<< 6))) || (dma_data_count[12] > package_length))begin
		// 	write_to_dma_enable				<= 12;
		// end
		// else if(((dma_remain_length[13] > 0) && (dma_remain_length[13] == (dma_data_count[13] <<< 6))) || (dma_data_count[13] > package_length))begin
		// 	write_to_dma_enable				<= 13;
		// end		
		// else if(((dma_remain_length[14] > 0) && (dma_remain_length[14] == (dma_data_count[14] <<< 6))) || (dma_data_count[14] > package_length))begin
		// 	write_to_dma_enable				<= 14;
		// end
		// else if(((dma_remain_length[15] > 0) && (dma_remain_length[15] == (dma_data_count[15] <<< 6))) || (dma_data_count[15] > package_length))begin
		// 	write_to_dma_enable				<= 15;
		// end
		else begin
			write_to_dma_enable				<= 5'b10000;
		end
	end


	always @(posedge clk)begin
		if(~rstn)begin
			current_addr 			        <= 1'b0;
		end
		else if(w_state == WRITE_INFO_CMD)begin
			current_addr					<= dma_recv_info_addr;
        end
		else if(w_state == WRITE_CMD)begin
			current_addr					<= session_addr[dma_buffer_id];
		end
		else begin
			current_addr					<= current_addr;
		end		
	end





	




	always @(posedge clk)begin
		if(~rstn)begin
			data_cnt 						<= 1'b0;
		end
		else if(axis_dma_write_data.last)begin
			data_cnt						<= 1'b0;
		end
		else if(axis_dma_write_data.ready & axis_dma_write_data.valid)begin
			data_cnt						<= data_cnt + 1'b1;
		end
		else begin
			data_cnt						<= data_cnt;
		end		
	end


	always @(posedge clk)begin
		if(~rstn)begin
			current_buffer_id 				<= 6'h3f;
		end
		else if(state == JUDGE)begin
			if({1'b0,current_session_id} == session_id[0])begin
				current_buffer_id				<= 0;
			end
			else if({1'b0,current_session_id} == session_id[1])begin
				current_buffer_id				<= 1;
			end
			else if({1'b0,current_session_id} == session_id[2])begin
				current_buffer_id				<= 2;
			end
			else if({1'b0,current_session_id} == session_id[3])begin
				current_buffer_id				<= 3;
			end
			else if({1'b0,current_session_id} == session_id[4])begin
				current_buffer_id				<= 4;
			end
			else if({1'b0,current_session_id} == session_id[5])begin
				current_buffer_id				<= 5;
			end
			else if({1'b0,current_session_id} == session_id[6])begin
				current_buffer_id				<= 6;
			end
			else if({1'b0,current_session_id} == session_id[7])begin
				current_buffer_id				<= 7;
			end
			// else if({1'b0,current_session_id} == session_id[8])begin
			// 	current_buffer_id				<= 8;
			// end
			// else if({1'b0,current_session_id} == session_id[9])begin
			// 	current_buffer_id				<= 9;
			// end
			// else if({1'b0,current_session_id} == session_id[10])begin
			// 	current_buffer_id				<= 10;
			// end
			// else if({1'b0,current_session_id} == session_id[11])begin
			// 	current_buffer_id				<= 11;
			// end
			// else if({1'b0,current_session_id} == session_id[12])begin
			// 	current_buffer_id				<= 12;
			// end
			// else if({1'b0,current_session_id} == session_id[13])begin
			// 	current_buffer_id				<= 13;
			// end
			// else if({1'b0,current_session_id} == session_id[14])begin
			// 	current_buffer_id				<= 14;
			// end
			// else if({1'b0,current_session_id} == session_id[15])begin
			// 	current_buffer_id				<= 15;
			// end
			// else if({1'b0,current_session_id} == session_id[16])begin
			// 	current_buffer_id				<= 16;
			// end
			// else if({1'b0,current_session_id} == session_id[17])begin
			// 	current_buffer_id				<= 17;
			// end
			// else if({1'b0,current_session_id} == session_id[18])begin
			// 	current_buffer_id				<= 18;
			// end
			// else if({1'b0,current_session_id} == session_id[19])begin
			// 	current_buffer_id				<= 19;
			// end
			// else if({1'b0,current_session_id} == session_id[20])begin
			// 	current_buffer_id				<= 20;
			// end
			// else if({1'b0,current_session_id} == session_id[21])begin
			// 	current_buffer_id				<= 21;
			// end
			// else if({1'b0,current_session_id} == session_id[22])begin
			// 	current_buffer_id				<= 22;
			// end
			// else if({1'b0,current_session_id} == session_id[23])begin
			// 	current_buffer_id				<= 23;
			// end
			// else if({1'b0,current_session_id} == session_id[24])begin
			// 	current_buffer_id				<= 24;
			// end
			// else if({1'b0,current_session_id} == session_id[25])begin
			// 	current_buffer_id				<= 25;
			// end
			// else if({1'b0,current_session_id} == session_id[26])begin
			// 	current_buffer_id				<= 26;
			// end
			// else if({1'b0,current_session_id} == session_id[27])begin
			// 	current_buffer_id				<= 27;
			// end
			// else if({1'b0,current_session_id} == session_id[28])begin
			// 	current_buffer_id				<= 28;
			// end
			// else if({1'b0,current_session_id} == session_id[29])begin
			// 	current_buffer_id				<= 29;
			// end
			// else if({1'b0,current_session_id} == session_id[30])begin
			// 	current_buffer_id				<= 30;
			// end
			// else if({1'b0,current_session_id} == session_id[31])begin
			// 	current_buffer_id				<= 31;
			// end
			else begin
				current_buffer_id				<= current_buffer_id;
			end
		end						
		else begin
			current_buffer_id				<= current_buffer_id;
		end		
	end
////////////////////////ACK RECV////////////////

	assign m_axis_conn_ack_recv.valid = state == ACK_RECV;
	assign m_axis_conn_ack_recv.data = {ctrl_data[16],current_buffer_id[4:0],current_session_id};

/////////////////////////////////////////////////

	always @(posedge clk)begin
		if(~rstn)begin
			state							<= IDLE;
		end
		else begin
			fifo_cmd_rd_en					<= 1'b0;
			case(state)				
				IDLE:begin
					if(~fifo_cmd_empty)begin
						fifo_cmd_rd_en		<= 1'b1;
						state				<= START;
					end
					else begin
						state				<= IDLE;
					end
                end
                START:begin
					if(fifo_cmd_rd_valid)begin
						state           	<= JUDGE;
						current_length		<= fifo_cmd_rd_data[31:16];
						current_session_id	<= fifo_cmd_rd_data[15:0];
						des_ip_addr			<= fifo_cmd_rd_data[63:32];
						des_port			<= fifo_cmd_rd_data[79:64];
						session_close_flag	<= fifo_cmd_rd_data[87:80];
					end
					else begin
						state				<= START;
					end
                end
				JUDGE:begin
					if(session_close_flag == 8'hff)begin
						ctrl_data			<= 4;
						state				<= INFO_WRITE;
					end
					else begin
						state				<= JUDGE1;
					end					
				end
				JUDGE1:begin
					if(current_length == 16'h0)begin
						state				<= IDLE;
					end
					else if(remain_length[current_buffer_id[4:0]] == 0)begin
						state				<= READ_CTRL;
					end
					else begin
						state				<= WRITE_CMD;
					end
				end
				READ_CTRL:begin
					if(s_axis_rx_data.valid && s_axis_rx_data.ready)begin
						ctrl_data			<= s_axis_rx_data.data[79:0];
						if(s_axis_rx_data.data[2:0] == 1)begin
							state			<= ACK_RECV;
						end
						else if(s_axis_rx_data.data[2:0] == 3)begin
							state			<= TCP_DATA_RECV;
						end
						else begin
							state			<= INFO_WRITE;
						end
					end
					else begin
						state				<= READ_CTRL;
					end
				end
				INFO_WRITE:begin
					state					<= INFO_WRITE_DATA;
				end
				INFO_WRITE_DATA:begin
					if(axis_tcp_info.ready && axis_tcp_info.valid)begin
						if(ctrl_data[2:0] == 4)begin
							state			<= JUDGE1;
						end
						else begin
							state			<= IDLE;
						end				
					end
					else begin
						state				<= INFO_WRITE_DATA;
					end
				end		
				ACK_RECV:begin
					if(m_axis_conn_ack_recv.ready && m_axis_conn_ack_recv.valid)begin
						state				<= IDLE;
					end
					else begin
						state				<= ACK_RECV;
					end
				end						
				TCP_DATA_RECV:begin
					state					<= IDLE;
				end											
				WRITE_CMD:begin
					state					<= WRITE_DATA;
				end
				// WRITE_CTRL_DATA:begin
				// 	if(axis_rx_data.ready & axis_rx_data.valid)begin
				// 		state				<= WRITE_DATA;
				// 	end	
				// 	else begin
				// 		state				<= WRITE_CTRL_DATA;
				// 	end                    
				// end				
				WRITE_DATA:begin
					if(s_axis_rx_data.valid & s_axis_rx_data.ready & s_axis_rx_data.last)begin
						state			<= END;
					end	
					else begin
						state			<= WRITE_DATA;
					end
                end
				END:begin
					state			<= IDLE;
				end
			endcase
		end
	end







	always @(posedge clk)begin
		if(~rstn)begin
			w_state							<= IDLE;
			dma_write_cmd_valid				<= 0;
		end
		else begin
			case(w_state)				
				IDLE:begin
					if(dma_info_count > 0)begin
						w_state				<= WRITE_INFO_CMD;
						current_data_count	<= 1;
						current_dma_length	<= 32'h40;
						data_minus			<= 0;
					end
					else if(write_to_dma_enable[4] == 0 && (dma_data_count[write_to_dma_enable[3:0]] > package_length) )begin
						w_state				<= WRITE_CMD;
						dma_buffer_id		<= write_to_dma_enable[3:0];
						current_data_count	<= package_length;
						current_dma_length	<= (package_length<<<6) + 32'h80;
						data_minus			<= package_length;
					end
					else if(write_to_dma_enable[4] == 0)begin
						w_state				<= WRITE_CMD;
						dma_buffer_id		<= write_to_dma_enable[3:0];
						current_data_count	<= dma_data_count[write_to_dma_enable[3:0]];
						current_dma_length	<= (dma_data_count[write_to_dma_enable[3:0]]<<<6) + 32'h80;
						data_minus			<= dma_data_count[write_to_dma_enable[3:0]];
					end
					else begin
						w_state				<= IDLE;
					end
                end											
				WRITE_INFO_CMD:begin
					dma_write_cmd_valid		<= 1'b1;
					if(axis_dma_write_cmd.valid && axis_dma_write_cmd.ready)begin
						dma_write_cmd_valid	<= 1'b0;
						w_state				<= WRITE_INFO_DATA;
					end
					else begin
						w_state				<= WRITE_INFO_CMD;
					end					
				end
				WRITE_INFO_DATA:begin
					if(axis_dma_write_data.ready & axis_dma_write_data.valid)begin
						w_state				<= END;
					end	
					else begin
						w_state				<= WRITE_INFO_DATA;
					end                    
				end	
				WRITE_CMD:begin
					dma_write_cmd_valid		<= 1'b1;
					if(axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
						dma_write_cmd_valid	<= 1'b0;
						w_state				<= WRITE_CTRL_DATA;
					end
					else begin
						w_state				<= WRITE_CMD;
					end
				end
				WRITE_CTRL_DATA:begin
					if(axis_dma_write_data.ready & axis_dma_write_data.valid)begin
						w_state				<= WRITE_DATA;
					end	
					else begin
						w_state				<= WRITE_CTRL_DATA;
					end                    
				end				
				WRITE_DATA:begin
					if((data_cnt == data_minus) && (axis_dma_write_data.ready & axis_dma_write_data.valid))begin
						w_state			<= WRITE_END_DATA;
					end	
					else begin
						w_state			<= WRITE_DATA;
					end
				end
				WRITE_END_DATA:begin
					if(axis_dma_write_data.ready & axis_dma_write_data.valid)begin
						w_state				<= END;
					end	
					else begin
						w_state				<= WRITE_END_DATA;
					end                    
				end					
				END:begin
					w_state			<= IDLE;
				end
			endcase
		end
	end

/////////////////////////////////DEBUG/////////////////////	

	reg 									wr_th_en;
	reg [31:0]								wr_th_sum;
	reg [31:0]								wr_data_cnt;
	reg [31:0]								wr_length_minus;

	always@(posedge clk)begin
		wr_length_minus							<= control_reg[7] -1;
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_th_en						<= 1'b0;
		end  
		else if(wr_length_minus == wr_data_cnt)begin
			wr_th_en						<= 1'b0;
		end
		else if((axis_dma_write_cmd.ready & axis_dma_write_cmd.valid) && (w_state==WRITE_CMD))begin
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
		else if((axis_dma_write_data.ready & axis_dma_write_data.valid) && (w_state==WRITE_DATA)) begin
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

/////////////////////////////////////////////////////////////	
	reg 									wr_tcp_en;
	reg [31:0]								wr_tcp_sum;
	reg [31:0]								wr_tcp_data_cnt;

	always@(posedge clk)begin
		if(~rstn)begin
			wr_tcp_en						<= 1'b0;
		end  
		else if(wr_length_minus == wr_tcp_data_cnt)begin
			wr_tcp_en						<= 1'b0;
		end
		else if((axis_dma_write_cmd.ready & axis_dma_write_cmd.valid) && (w_state==WRITE_CMD))begin
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
		else if((s_axis_rx_data.ready & s_axis_rx_data.valid) && (state==WRITE_DATA)) begin
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

	assign status_reg[1]					= wr_tcp_sum;	

/////////////////////////////////////////////////////////////




	ila_tx tx (
		.clk(clk), // input wire clk
	
	
		.probe0(axis_dma_write_cmd.valid), // input wire [0:0]  probe0  
		.probe1(axis_dma_write_cmd.ready), // input wire [0:0]  probe1 
		.probe2(axis_dma_write_cmd.address), // input wire [63:0]  probe2 
		.probe3(axis_dma_write_cmd.length), // input wire [31:0]  probe3 
		.probe4(axis_dma_write_data.valid), // input wire [0:0]  probe4 
		.probe5(axis_dma_write_data.ready), // input wire [0:0]  probe5 
		.probe6(axis_dma_write_data.last), // input wire [0:0]  probe6 
		.probe7(axis_dma_write_data.data[63:32]), // input wire [31:0]  probe7
		.probe8(axis_dma_write_data.keep), // input wire [63:0]  probe7
		.probe9(state), // input wire [4:0]  probe9 
		.probe10(s_axis_set_buffer_id.valid), // input wire [0:0]  probe10 
	   	.probe11(s_axis_set_buffer_id.ready), // input wire [0:0]  probe11 
	   	.probe12(current_dma_length), // input wire [87:0]  probe12
		.probe13(current_session_id), // input wire [31:0]  probe13 
		.probe14(current_buffer_id), // input wire [31:0]  probe14 
		.probe15(session_id[0]), // input wire [15:0]  probe15 
		.probe16(w_state) // input wire [15:0]  probe16
	);



endmodule
