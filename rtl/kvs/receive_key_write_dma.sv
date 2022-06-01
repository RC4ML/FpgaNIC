module receive_key_write_dma#(
    parameter       TIME_OUT_CYCLE =   32'h9502_F900// 32'h9502_F900
) ( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,

    //DMA Commands
    axis_mem_cmd.master         axis_dma_write_cmd,

    //DMA Data streams      
    axi_stream.master           axis_dma_write_data,

	//tcp recv   
    axis_meta.slave    			s_axis_rx_metadata,
    axi_stream.slave   			s_axis_rx_data,

	//control reg
	input wire[15:0][31:0]		control_reg,
	output wire[7:0][31:0]		status_reg

    );

	localparam [4:0]		IDLE 				= 5'h0,
							START				= 5'h1,
							INFO_WRITE			= 5'h2,
							INFO_WRITE_DATA		= 5'h3,
                            READ_LENGTH         = 5'h4,    
							WRITE_CMD			= 5'h5,
                            INFO_WRITE_TIMER    = 5'h9,    
							TIMER_CMD			= 5'ha,	                           							
							WRITE_CTRL_DATA		= 5'h6,
							WRITE_DATA			= 5'h7,
							END         		= 5'h8;



    reg [5:0]                               state,w_state;   

    reg [63:0]                              dma_base_addr;
	reg [7:0]								dma_info_cl_minus;
	reg [7:0]								info_cnt,dma_info_cnt;
	reg [15:0]								key_num;
    reg                                     wr_start,wr_start_r;
	reg [31:0]								dma_info_length;

	always @(posedge clk)begin	
        wr_start_r                          <= wr_start;
		dma_base_addr						<= {control_reg[1],control_reg[0]};
		dma_info_cl_minus					<= control_reg[2][7:0];
		key_num								<= control_reg[3][15:0];
		dma_info_length						<= (dma_info_cl_minus+1) <<< 6;
	end

	always @(posedge clk)begin
		if(~rstn)begin
			wr_start 			        	<= 1'b0;
		end
		else if(fifo_cmd_wr_en)begin
			wr_start						<= 1'b1;
        end
		else begin
			wr_start						<= wr_start;
		end		
	end


	//////////////notifications buffer ///////////
	wire 									fifo_cmd_wr_en;
	reg 									fifo_cmd_rd_en;			

	wire 									fifo_cmd_empty;	
	wire 									fifo_cmd_almostfull;		
	wire [87:0]								fifo_cmd_rd_data;
	wire 									fifo_cmd_rd_valid;
	wire [31:0]								fifo_cmd_count;

	assign s_axis_rx_metadata.ready 		= ~fifo_cmd_almostfull;
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
	.count      (fifo_cmd_count   )
	);

////////////////////////////////////////////////////////
//////////////dma information buffer ///////////

	axi_stream 								axis_tcp_info();
	axi_stream								axis_tcp_info_to_dma();
    reg[15:0][15:0]                         session_id,packet_length;
    reg[15:0]                               current_session_id;
    reg[31:0]                               total_packet_length;
    reg[3:0]                                packet_index;
	wire 									tcp_info_to_dma_last;	
	wire [31:0]								dma_info_count;	

	assign axis_tcp_info.keep				= 64'hffff_ffff_ffff_ffff;
	assign axis_tcp_info.last				= 1'b0;
	// assign axis_tcp_info.data				= {packet_length[0],session_id[0],packet_length[1],session_id[1],packet_length[2],session_id[2],packet_length[3],session_id[3],packet_length[4],session_id[4],packet_length[5],session_id[5],packet_length[6],session_id[6],packet_length[7],session_id[7],
	// 										packet_length[8],session_id[8],packet_length[9],session_id[9],packet_length[10],session_id[10],packet_length[11],session_id[11],packet_length[12],session_id[12],packet_length[13],session_id[13],packet_length[14],session_id[14],packet_length[15],session_id[15]};
	assign axis_tcp_info.data				= {key_num,session_id[0],key_num,session_id[1],key_num,session_id[2],key_num,session_id[3],key_num,session_id[4],key_num,session_id[5],key_num,session_id[6],key_num,session_id[7],
											key_num,session_id[8],key_num,session_id[9],key_num,session_id[10],key_num,session_id[11],key_num,session_id[12],key_num,session_id[13],key_num,session_id[14],key_num,session_id[15]};

	assign axis_tcp_info.valid				= (state == INFO_WRITE_DATA);// || (state == INFO_WRITE_TIMER) ;

	// assign axis_tcp_info_to_dma.ready 		= w_state == WRITE_CTRL_DATA;


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

      assign axis_tcp_info_to_dma.ready 		= axis_dma_write_data.ready && (w_state == WRITE_CTRL_DATA);

    genvar i;
    generate
        for(i = 0; i < 16; i = i + 1) begin
            always @(posedge clk) begin
                if(~rstn)begin
                    session_id[i]               <= 0;
                    packet_length[i]            <= 0;   
                end
                else if((state == WRITE_CMD))begin//|| (state == TIMER_CMD)
                    session_id[i]               <= 0;
                    packet_length[i]            <= 0;                    
                end
                else if((state == INFO_WRITE) && (packet_index == i))begin
                    session_id[i]               <= current_session_id;
                    packet_length[i]            <= current_length;
                end
                else begin
                    session_id[i]               <= session_id[i];
                    packet_length[i]            <= packet_length[i];   
                end
            end
  
        end
    endgenerate

    always @(posedge clk) begin
        if(~rstn)begin
            total_packet_length                 <= 0;
        end
        else if((state == INFO_WRITE) && (packet_index == 0) && (info_cnt == 0))begin
            total_packet_length                 <= current_length;
        end
        else if(state == INFO_WRITE)begin
            total_packet_length                 <= total_packet_length + current_length;
        end        
        else begin
            total_packet_length                 <= total_packet_length;
        end
    end


////////////////////////////////////////////////////////	
	//////////////length buffer ///////////
	wire 									fifo_length_wr_en;
	reg 									fifo_length_rd_en;			

	wire 									fifo_length_empty;	
	wire 									fifo_length_almostfull;		
	wire [31:0]								fifo_length_rd_data;
	wire 									fifo_length_rd_valid;	
	// wire [6:0]								fifo_length_count;


	assign fifo_length_wr_en					= ((state == WRITE_CMD) && (info_cnt == (dma_info_cl_minus+1))); // || (state == TIMER_CMD)) ;

	blockram_fifo #( 
		.FIFO_WIDTH      ( 32 ), //64 
		.FIFO_DEPTH_BITS ( 10 )  //determine the size of 16  13
    ) inst_length_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (fifo_length_wr_en     ), //or one cycle later...
	.din        (total_packet_length ),
	.almostfull (fifo_length_almostfull), //back pressure to  

	//reading side.....
	.re         (fifo_length_rd_en     ),
	.dout       (fifo_length_rd_data   ),
	.valid      (fifo_length_rd_valid	),
	.empty      (fifo_length_empty     ),
	.count      (   )
	);

////////////////////////////////////////////////////////



    axi_stream 								axis_dma_fifo_data();
    reg [31:0]                              data_cnt;
    reg [63:0]                              current_addr;
    reg [31:0]                              current_dma_length;
    reg [15:0]                              current_length;

    wire                                    s_rx_data_ready,s_rx_data_valid;


	always @(posedge clk)begin
		if(~rstn)begin
			current_addr 			        <= 1'b0;
		end
		else if(wr_start & ~wr_start_r)begin
			current_addr					<= dma_base_addr;
        end
		else if((w_state == WRITE_CMD) & axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
			current_addr					<= current_addr + current_dma_length;
		end
		else begin
			current_addr					<= current_addr;
		end		
	end

    assign	axis_dma_write_cmd.address	    = current_addr;
    assign	axis_dma_write_cmd.length	    = current_dma_length; 
    assign 	axis_dma_write_cmd.valid		= w_state == WRITE_CMD; 

	always @(posedge clk)begin
		if(~rstn)begin
			data_cnt 						<= 1'b0;
		end
		else if(axis_dma_write_data.last)begin
			data_cnt						<= 1'b0;
		end
		else if(axis_dma_write_data.ready & axis_dma_write_data.valid)begin
			data_cnt						<= data_cnt + 32'd64;
		end
		else begin
			data_cnt						<= data_cnt;
		end		
	end


    assign s_axis_rx_data.ready             = s_rx_data_ready & (state == WRITE_DATA);
    assign s_rx_data_valid                  = s_axis_rx_data.valid & (state == WRITE_DATA);


    axis_data_fifo_512_d32768 write_data_slice_fifo (
        .s_axis_aresetn(rstn),          // input wire s_axis_aresetn
        .s_axis_aclk(clk),                // input wire s_axis_aclk
        .s_axis_tvalid(s_rx_data_valid),            // input wire s_axis_tvalid
        .s_axis_tready(s_rx_data_ready),            // output wire s_axis_tready
        .s_axis_tdata(s_axis_rx_data.data),              // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_rx_data.keep),              // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_rx_data.last),              // input wire s_axis_tlast
        .m_axis_tvalid(axis_dma_fifo_data.valid),            // output wire m_axis_tvalid
        .m_axis_tready(axis_dma_fifo_data.ready),            // input wire m_axis_tready
        .m_axis_tdata(axis_dma_fifo_data.data),              // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_dma_fifo_data.keep),              // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_dma_fifo_data.last)              // output wire m_axis_tlast
        // .axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
        // .axis_rd_data_count()  // output wire [31 : 0] axis_rd_data_count
      );


  
    assign axis_dma_write_data.valid	    = ((w_state == WRITE_DATA) && axis_dma_fifo_data.valid) || ((w_state == WRITE_CTRL_DATA) && axis_tcp_info_to_dma.valid);
    assign axis_dma_write_data.keep		    = 64'hffff_ffff_ffff_ffff;
    assign axis_dma_write_data.data		    = (w_state == WRITE_DATA) ? axis_dma_fifo_data.data : axis_tcp_info_to_dma.data;
    assign axis_dma_write_data.last         = (data_cnt == (current_dma_length - 32'h40)) && (axis_dma_write_data.ready & axis_dma_write_data.valid);

    assign axis_dma_fifo_data.ready 		= axis_dma_write_data.ready && (w_state == WRITE_DATA);

///////////////////////timer//////////

    // reg[31:0]                               timer;


	// always @(posedge clk)begin
	// 	if(~rstn)begin
	// 		timer 						<= 1'b0;
	// 	end
	// 	else if((state == WRITE_CMD) || (state == TIMER_CMD))begin
	// 		timer						<= 1'b0;
	// 	end
	// 	else begin
	// 		timer						<= timer + 32'd1;
	// 	end		
	// end

//////////////////////////////////////fsm//////////                  


    always @(posedge clk)begin
		if(~rstn)begin
			state							<= IDLE;
            packet_index                    <= 4'b0;
			info_cnt						<= 0;
		end
		else begin
			fifo_cmd_rd_en					<= 1'b0;
			case(state)				
				IDLE:begin
					if(~fifo_cmd_empty) begin
						fifo_cmd_rd_en		<= 1'b1;
						state				<= START;
					end
                    // else if((timer == TIME_OUT_CYCLE) && (packet_index != 0))begin
                    //     state				<= INFO_WRITE_TIMER;
                    // end
					else begin
						state				<= IDLE;
					end
                end
                START:begin
					if(fifo_cmd_rd_valid)begin
						state           	<= INFO_WRITE;
						current_length		<= fifo_cmd_rd_data[31:16];
						current_session_id	<= fifo_cmd_rd_data[15:0];
					end
					else begin
						state				<= START;
					end
                end
				INFO_WRITE:begin
                    packet_index            <= packet_index + 1'b1;
                    if(packet_index == 4'hf)begin
                        state				<= INFO_WRITE_DATA;
                    end
                    else begin
                        state				<= WRITE_DATA;
                    end
				end
				INFO_WRITE_DATA:begin
					if(axis_tcp_info.ready && axis_tcp_info.valid)begin
						info_cnt 			<= info_cnt + 1;
						// if(info_cnt == dma_info_cl_minus)begin							
						// 	state		   	<= WRITE_CMD;	
						// end
						// else begin
							state			<= WRITE_CMD;	
						// end		
					end
					else begin
						state				<= INFO_WRITE_DATA;
					end
				end	
				WRITE_CMD:begin
					if(info_cnt == (dma_info_cl_minus+1))begin
						info_cnt 			<= 0;
					end
					state					<= WRITE_DATA;
				end	                
				// INFO_WRITE_TIMER:begin
                //     packet_index            <= 0;
				// 	if(axis_tcp_info.ready && axis_tcp_info.valid)begin
				// 		state			    <= TIMER_CMD;			
				// 	end
				// 	else begin
				// 		state				<= INFO_WRITE_TIMER;
				// 	end
				// end                                
				// TIMER_CMD:begin
				// 	state					<= END;
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
            fifo_length_rd_en               <= 0; 
            current_dma_length              <= 0;
			dma_info_cnt					<= 0;
		end
		else begin
			fifo_length_rd_en               <= 0; 
			case(w_state)				
				IDLE:begin
					if(~fifo_length_empty)begin
						w_state				<= READ_LENGTH;
                        fifo_length_rd_en   <= 1;						
					end
					else begin
						w_state				<= IDLE;
					end
                end		
                READ_LENGTH:begin
                    if(fifo_length_rd_valid)begin
                        current_dma_length	<= fifo_length_rd_data + dma_info_length;
                        w_state				<= WRITE_CMD;
                    end
                    else begin
                        w_state				<= READ_LENGTH;
                    end
                end									
				WRITE_CMD:begin
					if(axis_dma_write_cmd.ready & axis_dma_write_cmd.valid)begin
						w_state				<= WRITE_CTRL_DATA;
					end
					else begin
						w_state				<= WRITE_CMD;
					end
				end
				WRITE_CTRL_DATA:begin
					if(axis_dma_write_data.ready & axis_dma_write_data.valid)begin
						dma_info_cnt		<= dma_info_cnt + 1;
						if(dma_info_cnt == dma_info_cl_minus)begin
							dma_info_cnt	<= 0;
							w_state			<= WRITE_DATA;
						end
						else begin
							w_state			<= WRITE_CTRL_DATA;
						end
					end	
					else begin
						w_state				<= WRITE_CTRL_DATA;
					end                    
				end				
				WRITE_DATA:begin
					if((data_cnt == (current_dma_length - 32'h40)) && (axis_dma_write_data.ready & axis_dma_write_data.valid))begin
						w_state			<= END;
					end	
					else begin
						w_state			<= WRITE_DATA;
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
	reg [31:0]								meta_cnt;

	always@(posedge clk)begin
		wr_length_minus							<= control_reg[4] -1;
	end

	always@(posedge clk)begin
		if(~rstn)begin
			wr_th_en						<= 1'b0;
		end  
		else if(wr_length_minus == wr_data_cnt)begin
			wr_th_en						<= 1'b0;
		end
		else if(s_axis_rx_metadata.ready & s_axis_rx_metadata.valid)begin
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


	always@(posedge clk)begin
		if(~rstn)begin
			meta_cnt						<= 1'b0;
		end  
		else if((s_axis_rx_metadata.ready & s_axis_rx_metadata.valid)) begin
			meta_cnt						<= meta_cnt + 1'b1;
		end		
		else begin
			meta_cnt						<= meta_cnt;
		end
	end


	assign status_reg[1]					= wr_th_sum;

	assign status_reg[0] = 					dma_info_count;
	assign status_reg[2] = 					fifo_cmd_count;
	assign status_reg[3] = 					meta_cnt;

	ila_kvs_s recv (
		.clk(clk), // input wire clk
	
	
		.probe0(state), // input wire [4:0]  probe0  
		.probe1(w_state), // input wire [4:0]  probe1
		.probe2(axis_dma_write_data.valid), // input wire [0:0]  probe2 
		.probe3(axis_dma_write_data.ready), // input wire [0:0]  probe3 
		.probe4(axis_dma_write_data.last), // input wire [0:0]  probe4 
		.probe5(axis_dma_write_data.data), // input wire [511:0]  probe5	
		.probe6(axis_dma_write_cmd.valid), // input wire [0:0]  probe6 
		.probe7(axis_dma_write_cmd.ready), // input wire [0:0]  probe7 
		.probe8(axis_dma_write_cmd.address), // input wire [63:0]  probe8 
		.probe9(axis_dma_write_cmd.length) // input wire [31:0]  probe9		
		
	);





endmodule