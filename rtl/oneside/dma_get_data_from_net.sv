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

module dma_get_data_from_net( 

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

	//control cmd
	axis_meta.master			m_axis_put_data_to_net, 
	//control reg
	output wire 				recv_done,	
	input wire[15:0][31:0]		control_reg,
	output wire[7:0][31:0]		status_reg

    );
    

///////////////////////dma_ex debug//////////

	localparam [4:0]		IDLE 				= 4'h0,
							START				= 4'h1,
							JUDGE				= 4'h2,
							READ_CTRL			= 4'h3,
							PUT_DATA			= 4'h4,
							GET_DATA			= 4'h5,
							WRITE_DATA			= 4'h6,
							END         		= 4'h7;	
				

	reg [3:0]								state;							

	reg [63:0]            					dma_base_addr;                       		
	reg [31:0]								remain_length;


    reg [63:0]                              current_addr;
	reg [15:0]                              current_length;
	reg [15:0]								current_session_id;
	reg [111:0]								ctrl_data;

	reg[31:0]								current_dma_length;	


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

///////////////////////////////////////////////////////

/////////////////////// //write data to dma
	wire 									axis_rx_data_ready;
	wire 									axis_rx_data_valid;
	wire 									axis_rx_data_last;
	wire [511:0]							axis_rx_data_data;
	wire [63:0]								axis_rx_data_keep;


	
	always @(posedge clk)begin
		if(~rstn)begin
			current_addr 			        <= 1'b0;
		end
		else if(state == READ_CTRL)begin
			current_addr					<= ctrl_data[79:48] + dma_base_addr;
		end
		else begin
			current_addr					<= current_addr;
		end		
	end

	always @(posedge clk)begin
		if(~rstn)begin
			current_dma_length 			        <= 1'b0;
		end
		else if(state == READ_CTRL)begin
			current_dma_length					<= s_axis_rx_data.data[47:16];
		end
		else begin
			current_dma_length					<= current_dma_length;
		end		
	end



	assign	axis_dma_write_cmd.address	    = current_addr;
	assign	axis_dma_write_cmd.length	    = current_dma_length; 
	assign 	axis_dma_write_cmd.valid		= state == GET_DATA; 
		

	always @(posedge clk)begin
		if(~rstn)begin
			remain_length 			    <= 1'b0;
		end
		else if(state == GET_DATA)begin
			remain_length                <= ctrl_data[47:16];
		end
		else if(axis_rx_data_valid & axis_rx_data_ready)begin
			remain_length				<= remain_length - 32'h40;
		end
		else begin
			remain_length				<= remain_length;
		end		
	end	

	assign s_axis_rx_data.ready 		= (state == READ_CTRL) ? 1 : ((state == WRITE_DATA) ? axis_rx_data_ready : 0);
	assign axis_rx_data_valid 			= (state == WRITE_DATA) ? s_axis_rx_data.valid : 0;
	assign axis_rx_data_data 			= s_axis_rx_data.data;
	assign axis_rx_data_keep			= s_axis_rx_data.keep;
	assign axis_rx_data_last			= (remain_length == 32'h40) && axis_rx_data_ready & axis_rx_data_valid;
	axis_data_fifo_512_d4096 write_data_slice_fifo (
		.s_axis_aresetn(rstn),          // input wire s_axis_aresetn
		.s_axis_aclk(clk),                // input wire s_axis_aclk
		.s_axis_tvalid(axis_rx_data_valid),            // input wire s_axis_tvalid
		.s_axis_tready(axis_rx_data_ready),            // output wire s_axis_tready
		.s_axis_tdata(axis_rx_data_data),              // input wire [511 : 0] s_axis_tdata
		.s_axis_tkeep(axis_rx_data_keep),              // input wire [63 : 0] s_axis_tkeep
		.s_axis_tlast(axis_rx_data_last),              // input wire s_axis_tlast
		.m_axis_tvalid(axis_dma_write_data.valid),            // output wire m_axis_tvalid
		.m_axis_tready(axis_dma_write_data.ready),            // input wire m_axis_tready
		.m_axis_tdata(axis_dma_write_data.data),              // output wire [511 : 0] m_axis_tdata
		.m_axis_tkeep(axis_dma_write_data.keep),              // output wire [63 : 0] m_axis_tkeep
		.m_axis_tlast(axis_dma_write_data.last),              // output wire m_axis_tlast
		.axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
		.axis_rd_data_count()  // output wire [31 : 0] axis_rd_data_count
	  );



	always @(posedge clk)begin	
		dma_base_addr						<= {control_reg[1],control_reg[0]};
	end


////////////////////////ACK RECV////////////////

	assign m_axis_put_data_to_net.valid = state == PUT_DATA;
	assign m_axis_put_data_to_net.data = {current_session_id,ctrl_data[111:16]};

/////////////////////////////////////////////////

	always @(posedge clk)begin
		if(~rstn)begin
			state							<= IDLE;
			fifo_cmd_rd_en					<= 1'b0;
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
						current_session_id	<= fifo_cmd_rd_data[15:0];
						current_length		<= fifo_cmd_rd_data[31:16];
					end
					else begin
						state				<= START;
					end
                end
				JUDGE:begin
					if(current_length == 16'h0)begin
						state				<= IDLE;
					end
					else if(remain_length == 0)begin
						state				<= READ_CTRL;
					end
					else begin
						state				<= WRITE_DATA;
					end
				end
				READ_CTRL:begin
					if(s_axis_rx_data.valid && s_axis_rx_data.ready)begin
						ctrl_data			<= s_axis_rx_data.data[111:0];
						if(s_axis_rx_data.data[2:0] == 4)begin
							state			<= PUT_DATA;
						end
						else if(s_axis_rx_data.data[2:0] == 5)begin
							state			<= GET_DATA;
						end
					end
					else begin
						state				<= READ_CTRL;
					end
				end	
				PUT_DATA:begin
					if(m_axis_put_data_to_net.ready && m_axis_put_data_to_net.valid)begin
						state				<= IDLE;
					end
					else begin
						state				<= PUT_DATA;
					end
				end						
				GET_DATA:begin
					if(axis_dma_write_cmd.ready && axis_dma_write_cmd.valid)begin
						state				<= WRITE_DATA;
					end
					else begin
						state				<= GET_DATA;
					end
				end															
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

///////////////////////////////////////////debug//////////////
	reg[31:0]								one_side_write_cnt;

	assign recv_done						= axis_dma_write_data.last & axis_dma_write_data.valid & axis_dma_write_data.ready;

	always@(posedge clk)begin
		if(~rstn)begin
			one_side_write_cnt				<= 0;
		end
		else if(recv_done)begin
			one_side_write_cnt				<= one_side_write_cnt + 1'b1;
		end
		else begin
			one_side_write_cnt				<= one_side_write_cnt;
		end
	end

	assign status_reg 						= one_side_write_cnt;

	ila_oneside_get ila_oneside_get (
		.clk(clk), // input wire clk
	
	
		.probe0(state), // input wire [3:0]  probe0  
		.probe1(remain_length), // input wire [31:0]  probe1 
		.probe2(m_axis_put_data_to_net.valid), // input wire [0:0]  probe2 
		.probe3(m_axis_put_data_to_net.ready), // input wire [0:0]  probe3 
		.probe4(m_axis_put_data_to_net.data), // input wire [127:0]  probe4 
		.probe5(axis_dma_write_cmd.valid), // input wire [0:0]  probe5 
		.probe6(axis_dma_write_cmd.ready), // input wire [0:0]  probe6 
		.probe7(axis_dma_write_cmd.address), // input wire [63:0]  probe7 
		.probe8(axis_dma_write_cmd.length), // input wire [31:0]  probe8 
		.probe9(axis_dma_write_data.valid), // input wire [0:0]  probe9 
		.probe10(axis_dma_write_data.ready), // input wire [0:0]  probe10 
		.probe11(axis_dma_write_data.last), // input wire [0:0]  probe11 
		.probe12(axis_dma_write_data.data[31:0]), // input wire [31:0]  probe12 
		.probe13(s_axis_rx_metadata.valid), // input wire [0:0]  probe13 
		.probe14(s_axis_rx_metadata.ready), // input wire [0:0]  probe14 
		.probe15(s_axis_rx_metadata.data), // input wire [87:0]  probe15 
		.probe16(s_axis_rx_data.valid), // input wire [0:0]  probe16 
		.probe17(s_axis_rx_data.ready), // input wire [0:0]  probe17 
		.probe18(s_axis_rx_data.last), // input wire [0:0]  probe18 
		.probe19(s_axis_rx_data.data[31:0]) // input wire [31:0]  probe19
	);

	
endmodule
