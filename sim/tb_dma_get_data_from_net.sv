`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/21/2020 02:05:24 PM
// Design Name: 
// Module Name: tb_dma_write_data
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


module tb_dma_get_data_from_net(

    );
    reg clk,rstn;

// DMA Signals
    axis_mem_cmd    axis_dma_write_cmd();
    axi_stream      axis_dma_write_data();
    
    axis_meta #(.WIDTH(88))     axis_tcp_rx_meta();
    axi_stream #(.WIDTH(512))    axis_tcp_rx_data();   

    axis_meta #(.WIDTH(112))     axis_put_data_to_net();
    
    assign axis_put_data_to_net.ready = 1;
    
    reg[15:0][31:0]     fpga_control_reg;
    wire[1:0][31:0]     fpga_status_reg; 
    

    reg[15:0][31:0]     fpga_control_reg;
    wire[1:0][31:0]     fpga_status_reg; 

    reg                 notification_valid;
    reg [87:0]          notification_data;
    reg [511:0]         rx_data_data;
    reg                 rx_data_valid;
    reg                 rx_data_last;
    reg [15:0]          rx_length;


	localparam [4:0]		IDLE 				= 4'h0,
							START				= 4'h1;
				

	reg [3:0]								state;	


    initial begin
        clk = 1'b1;
        rstn = 1'b0;
        fpga_control_reg[0] = 32'h1234_0000;
        fpga_control_reg[1] = 32'h0001_5678;
        notification_valid = 0;
        notification_data = 0;
        #1000
        rstn = 1'b1;
        #100
        notification_valid = 1;
        notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h40,16'h02};
        #10
        notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h40,16'h02};
        #10
        // notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h40,16'h02};
        // #10
        // notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h40,16'h02};
        // #10
        notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h400,16'h02};  
        #160
        notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h40,16'h02};
        #10
        notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h400,16'h02};  
        #160        
        notification_data = {8'hff,16'h1234,32'h010bd1d4,16'h0,16'h02}; 
              
        #10
        notification_valid = 0;

    end

    always #5 clk = ~clk;

    reg [31:0]  cnt;

    reg                     fifo_cmd_rd_en;
    wire[15:0]              fifo_cmd_rd_data;
    wire                    fifo_cmd_rd_valid;
    wire                    fifo_cmd_empty;


	blockram_fifo #( 
		.FIFO_WIDTH      ( 16 ), //64 
		.FIFO_DEPTH_BITS ( 10 )  //determine the size of 16  13
	) inst_tcp_notice_fifo (
	.clk        (clk),
	.reset_n    (rstn),

	//Writing side....
	.we         (notification_valid    ), //or one cycle later...
	.din        (notification_data[31:16] ),
	.almostfull (), //back pressure to  

	//reading side.....
	.re         (fifo_cmd_rd_en     ),
	.dout       (fifo_cmd_rd_data   ),
	.valid      (fifo_cmd_rd_valid	),
	.empty      (fifo_cmd_empty     ),
	.count      (   )
	);

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
                    if(axis_tcp_rx_data.last)begin
                        state               <= IDLE;
                    end
                    else begin
                        state               <= START;
                    end
                end
			endcase
		end
	end

    always@(posedge clk)begin
        if(~rstn)begin
            rx_data_data <= {400'b0,32'h2345,32'h3456,32'h4000,16'h4};
        end
        else if(rx_data_data[31:0] == 32'h4000_0005)begin
            rx_data_data <= rx_data_data;
        end
        else if(axis_tcp_rx_data.ready && axis_tcp_rx_data.valid)begin
            rx_data_data <= rx_data_data +1'b1;
        end
        else begin
            rx_data_data <= rx_data_data;
        end
    end



    always@(posedge clk)begin
        if(~rstn)begin
            rx_length <= 1'b0;
        end
        else if(fifo_cmd_rd_valid)begin
            rx_length <= fifo_cmd_rd_data;
        end
        else begin
            rx_length <= rx_length;
        end
    end


    always@(posedge clk)begin
        if(~rstn)begin
            cnt <= 1'b0;
        end
        else if(axis_tcp_rx_data.last)begin
            cnt <= 0;
        end
        else if(axis_tcp_rx_data.ready && axis_tcp_rx_data.valid)begin
            cnt <= cnt +1'b1;
        end
        else begin
            cnt <= cnt;
        end
    end

    assign axis_dma_write_cmd.ready = 1'b1;
    assign axis_dma_write_data.ready = 1'b1;

    assign axis_tcp_rx_data.valid = state == START;
    assign axis_tcp_rx_data.data = rx_data_data;
    assign axis_tcp_rx_data.keep = 64'hffff_ffff_ffff_ffff;
    assign axis_tcp_rx_data.last = (cnt == ((rx_length >> 6) - 1)) && axis_tcp_rx_data.ready && axis_tcp_rx_data.valid;


    assign axis_tcp_rx_meta.valid = notification_valid;
    assign axis_tcp_rx_meta.data = notification_data;


    dma_get_data_from_net dma_get_data_from_net_inst( 
    
        //user clock input
        .clk                        (clk),
        .rstn                       (rstn),
    
        //DMA Commands
        .axis_dma_write_cmd         (axis_dma_write_cmd),
    
        //DMA Data streams      
        .axis_dma_write_data        (axis_dma_write_data),
    
        //tcp send        
        .s_axis_rx_metadata         (axis_tcp_rx_meta),
        .s_axis_rx_data             (axis_tcp_rx_data),
    
        //control reg
        .m_axis_put_data_to_net     (axis_put_data_to_net),
        .control_reg                (fpga_control_reg),
        .status_reg                 (fpga_status_reg)
    
        );

endmodule
