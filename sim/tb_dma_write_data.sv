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


module tb_dma_write_data(

    );
    reg clk,rstn;

// DMA Signals
    axis_mem_cmd    axis_dma_write_cmd();
    axi_stream      axis_dma_write_data();
    
    axis_meta #(.WIDTH(88))     axis_tcp_notification();
    axis_meta #(.WIDTH(32))     axis_tcp_read_pkg();
    
    axis_meta #(.WIDTH(16))     axis_tcp_rx_meta();
    axi_stream #(.WIDTH(512))    axis_tcp_rx_data();   

    axis_meta #(.WIDTH(22))     conn_ack_recv();
    axis_meta #(.WIDTH(21))     set_buffer_id();
    
    assign set_buffer_id.valid = 1;
    assign set_buffer_id.data = 16'h2;
    
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


    initial begin
        clk = 1'b1;
        rstn = 1'b0;
        fpga_control_reg[0] = 32'h1234_0000;
        fpga_control_reg[1] = 32'h0001_5678;
        fpga_control_reg[2] = 32'h800_0000;
        fpga_control_reg[3] = 32'h40000;
        fpga_control_reg[4] = 32'h40000;
        fpga_control_reg[5] = 32'h8;
        fpga_control_reg[6] = 32'd32;
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
        notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h40,16'h02};
        #10
        notification_data = {8'h0,16'h1234,32'h010bd1d4,16'h40,16'h02};
        #10
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
	.we         (axis_tcp_read_pkg.ready & axis_tcp_read_pkg.valid     ), //or one cycle later...
	.din        (axis_tcp_read_pkg.data[31:16] ),
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
                    
                end
			endcase
		end
	end


    always@(posedge clk)begin
        if(~rstn)begin
            rx_data_valid <= 1'b0;
        end
        else if(axis_tcp_read_pkg.ready & axis_tcp_read_pkg.valid)begin
            rx_data_valid <= 1'b1;
        end        
        else if(axis_tcp_rx_data.last)begin
            rx_data_valid <= 1'b0;
        end
        else begin
            rx_data_valid <= rx_data_valid;
        end
    end

    always@(posedge clk)begin
        if(~rstn)begin
            rx_data_data <= {480'b0,16'h4000,16'b0};
        end
        else if(rx_data_data == 32'h4000_0003)begin
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

    assign axis_tcp_rx_data.valid = rx_data_valid;
    assign axis_tcp_rx_data.data = rx_data_data;
    assign axis_tcp_rx_data.keep = 64'hffff_ffff_ffff_ffff;
    assign axis_tcp_rx_data.last = (cnt == ((rx_length >> 6) - 1)) && axis_tcp_rx_data.ready && axis_tcp_rx_data.valid;


    assign axis_tcp_notification.valid = notification_valid;
    assign axis_tcp_notification.data = notification_data;

    assign axis_tcp_read_pkg.ready = 1;

    assign conn_ack_recv.ready = 1'b1;

    dma_write_data_from_tcp dma_write_data_inst( 
    
        //user clock input
        .clk                        (clk),
        .rstn                       (rstn),
    
        //DMA Commands
        .axis_dma_write_cmd         (axis_dma_write_cmd),
    
        //DMA Data streams      
        .axis_dma_write_data        (axis_dma_write_data),
    
        //tcp send
        .s_axis_notifications       (axis_tcp_notification),
        .m_axis_read_package        (axis_tcp_read_pkg),
        
        .s_axis_rx_metadata         (axis_tcp_rx_meta),
        .s_axis_rx_data             (axis_tcp_rx_data),
    
        //control reg
        .s_axis_set_buffer_id              (set_buffer_id),
        .m_axis_conn_ack_recv              (conn_ack_recv),
        .control_reg                (fpga_control_reg),
        .status_reg                 (fpga_status_reg)
    
        );

endmodule
