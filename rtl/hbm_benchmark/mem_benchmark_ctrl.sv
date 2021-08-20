`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
/*
 * Copyright 2019 - 2020, RC4ML, Zhejiang University
 *
 * This hardware operator is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//////////////////////////////////////////////////////////////////////////////////


module mem_benchmark_ctrl#(
	parameter N_MEM_INTF			      = 32
)(
    input                               clk,
    input                               rstn,

    input     		                    hbm_clk,
    input                               hbm_rstn,

    output reg[N_MEM_INTF-1:0][511:0]   lt_params,     

///////////hbm——test——reg
	input                             lat_timer_valid , //log down lat_timer when lat_timer_valid is 1. 
    input         [15:0]              lat_timer,
/////////////////////////
    input[15:0][31:0]                  fpga_control,
    output[31:0]                       fpga_status

      
    );

    reg [5:0]                         wr_addr;
    reg [31:0]                        wr_data;

    reg [5:0]                         rd_addr;

    reg [11:0]                        addra_ori;
    reg [11:0]                        addrb_ori;

    wire[15:0]				          doutb;
    reg [15:0]                        lat_timer_out;                                  
    reg                               ram_wr_en;


    reg       [ 31:0]             work_group_size;
    reg       [ 31:0]             stride;
    reg       [ 63:0]             num_mem_ops;
    reg       [ 31:0]             mem_burst_size;
    reg       [ 33:0]             initial_addr;
    reg       [  7:0]             hbm_channel;
    reg                           latency_test_enable;



  always @(posedge hbm_clk)begin
    work_group_size              <= fpga_control[ 1];
    stride                       <= fpga_control[ 2];
    num_mem_ops[31: 0]           <= fpga_control[ 3];
    num_mem_ops[63:32]           <= fpga_control[ 4];
    mem_burst_size               <= fpga_control[ 5];
    initial_addr                 <= {fpga_control[ 6],2'b00};
    latency_test_enable          <= fpga_control[9][0];
    hbm_channel                  <= fpga_control[10][7:0];
  end

  always @(posedge clk)begin
    addrb_ori                    <= fpga_control[12][11:0];
  end  

  
//generate end generate
  genvar i;
  // Instantiate engines
  generate
  for(i = 0; i < N_MEM_INTF; i++) 
  begin
  
      always @(posedge hbm_clk) begin
          if(~hbm_rstn)begin
              lt_params[i][ 31:  0]       <=  work_group_size;
              lt_params[i][ 63: 32]       <=  stride;
              lt_params[i][127: 64]       <=  num_mem_ops;
              lt_params[i][159:128]       <=  mem_burst_size;
              lt_params[i][33+159:160]    <=  32'h1000_0000 * i;
              lt_params[i][224]           <=  1'b0;
              lt_params[i][225+:5]        <=  i;
              lt_params[i][511:256]       <=  lt_params[i][255:0];
          end
          else if(hbm_channel == i)begin
              lt_params[i][ 31:  0]       <=  work_group_size;
              lt_params[i][ 63: 32]       <=  stride;
              lt_params[i][127: 64]       <=  num_mem_ops;
              lt_params[i][159:128]       <=  mem_burst_size;
              lt_params[i][33+159:160]    <=  initial_addr;
              lt_params[i][224]           <=  latency_test_enable;
              lt_params[i][225+:5]        <=  i;
              lt_params[i][511:256]       <=  lt_params[i][255:0];
          end
          else begin 
              lt_params[i]                <=  lt_params[i];
          end
      end
  end
  endgenerate


	assign	    fpga_status             = {16'b0,lat_timer_out};





  wire [15:0]                     fifo_data_out;
  reg [15:0]                      ram_data_in;
  wire                            fifo_empty;
  wire                            fifo_rd_en;


	always@(posedge clk)begin
		if(~rstn)begin
			addra_ori					    <= 0;
        end
        else if(addra_ori == 12'hfff)begin
            addra_ori   					<= addra_ori;
        end
		else if(fifo_rd_en)begin
			addra_ori   					<= addra_ori + 1'b1;
		end
		else begin
			addra_ori   					<= addra_ori;
		end
	end



  assign fifo_rd_en = ~fifo_empty;

  reg[15:0] lat_timer_i;
  reg       lat_timer_valid_i;
  always @(posedge hbm_clk)begin
    lat_timer_i                       <= lat_timer;
    lat_timer_valid_i                 <= lat_timer_valid;
  end

indep_fwft_fifo_w16 lat_time_pri_fifo (
  .rst(~hbm_rstn),                  // input wire rst
  .wr_clk(hbm_clk),            // input wire wr_clk
  .rd_clk(clk),            // input wire rd_clk
  .din(lat_timer_i),                  // input wire [15 : 0] din
  .wr_en(lat_timer_valid_i),              // input wire wr_en
  .rd_en(fifo_rd_en),              // input wire rd_en
  .dout(fifo_data_out),                // output wire [15 : 0] dout
  .full(),                // output wire full
  .empty(fifo_empty),              // output wire empty
  .wr_rst_busy(),  // output wire wr_rst_busy
  .rd_rst_busy()  // output wire rd_rst_busy
);


  always @(posedge clk)begin
    if(~rstn)begin
      lat_timer_out                 <= 8'b0;
    end
    else begin
      lat_timer_out                 <= doutb;
    end
  end

  always @(posedge clk)begin
    if(~rstn)begin
      ram_wr_en               <= 1'b0;
    end
    else begin
      ram_wr_en                <= fifo_rd_en;
    end                  
  end

  always @(posedge clk)begin
    ram_data_in                   <= fifo_data_out;
  end

   xpm_memory_tdpram #(
      .ADDR_WIDTH_A(12),               // DECIMAL
      .ADDR_WIDTH_B(12),               // DECIMAL
      .AUTO_SLEEP_TIME(0),            // DECIMAL
      .BYTE_WRITE_WIDTH_A(16),        // DECIMAL
      .BYTE_WRITE_WIDTH_B(16),        // DECIMAL
      .CASCADE_HEIGHT(0),             // DECIMAL
      .CLOCKING_MODE("common_clock"), // String
      .ECC_MODE("no_ecc"),            // String
      .MEMORY_INIT_FILE("none"),      // String
      .MEMORY_INIT_PARAM("0"),        // String
      .MEMORY_OPTIMIZATION("true"),   // String
      .MEMORY_PRIMITIVE("ultra"),      // String
      .MEMORY_SIZE(65536),             // DECIMAL
      .MESSAGE_CONTROL(0),            // DECIMAL
      .READ_DATA_WIDTH_A(16),         // DECIMAL
      .READ_DATA_WIDTH_B(16),         // DECIMAL
      .READ_LATENCY_A(2),             // DECIMAL
      .READ_LATENCY_B(2),             // DECIMAL
      .READ_RESET_VALUE_A("0"),       // String
      .READ_RESET_VALUE_B("0"),       // String
      .RST_MODE_A("SYNC"),            // String
      .RST_MODE_B("SYNC"),            // String
      .SIM_ASSERT_CHK(0),             // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
      .USE_EMBEDDED_CONSTRAINT(0),    // DECIMAL
      .USE_MEM_INIT(1),               // DECIMAL
      .WAKEUP_TIME("disable_sleep"),  // String
      .WRITE_DATA_WIDTH_A(16),        // DECIMAL
      .WRITE_DATA_WIDTH_B(16),        // DECIMAL
      .WRITE_MODE_A("no_change"),     // String
      .WRITE_MODE_B("no_change")      // String
   )
   xpm_memory_tdpram_inst (
      .dbiterra(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                       // on the data output of port A.

      .dbiterrb(),             // 1-bit output: Status signal to indicate double bit error occurrence
                                       // on the data output of port A.
            
      .douta(),                   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
      .doutb(doutb),                   // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
      .sbiterra(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                       // on the data output of port A.

      .sbiterrb(),             // 1-bit output: Status signal to indicate single bit error occurrence
                                       // on the data output of port B.

      .addra(addra_ori),                   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
      .addrb(addrb_ori),                   // ADDR_WIDTH_B-bit input: Address for port B write and read operations.
      .clka(clk),                     // 1-bit input: Clock signal for port A. Also clocks port B when
                                       // parameter CLOCKING_MODE is "common_clock".

      .clkb(clk),                     // 1-bit input: Clock signal for port B when parameter CLOCKING_MODE is
                                       // "independent_clock". Unused when parameter CLOCKING_MODE is
                                       // "common_clock".

      .dina(ram_data_in),                     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
      .dinb(0),                     // WRITE_DATA_WIDTH_B-bit input: Data input for port B write operations.
      .ena(1),                       // 1-bit input: Memory enable signal for port A. Must be high on clock
                                       // cycles when read or write operations are initiated. Pipelined
                                       // internally.

      .enb(1),                       // 1-bit input: Memory enable signal for port B. Must be high on clock
                                       // cycles when read or write operations are initiated. Pipelined
                                       // internally.

      .injectdbiterra(1'b0), // 1-bit input: Controls double bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .injectdbiterrb(1'b0), // 1-bit input: Controls double bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .injectsbiterra(1'b0), // 1-bit input: Controls single bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .injectsbiterrb(1'b0), // 1-bit input: Controls single bit error injection on input data when
                                       // ECC enabled (Error injection capability is not available in
                                       // "decode_only" mode).

      .regcea(1'b1),                 // 1-bit input: Clock Enable for the last register stage on the output
                                       // data path.

      .regceb(1'b1),                 // 1-bit input: Clock Enable for the last register stage on the output
                                       // data path.

      .rsta(~rstn),                     // 1-bit input: Reset signal for the final port A output register stage.
                                       // Synchronously resets output port douta to the value specified by
                                       // parameter READ_RESET_VALUE_A.

      .rstb(~rstn),                     // 1-bit input: Reset signal for the final port B output register stage.
                                       // Synchronously resets output port doutb to the value specified by
                                       // parameter READ_RESET_VALUE_B.

      .sleep(1'b0),                   // 1-bit input: sleep signal to enable the dynamic power saving feature.
      .wea(ram_wr_en),                       // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
                                       // for port A input data port dina. 1 bit wide when word-wide writes are
                                       // used. In byte-wide write configurations, each bit controls the
                                       // writing one byte of dina to address addra. For example, to
                                       // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
                                       // is 32, wea would be 4'b0010.

      .web(0)                        // WRITE_DATA_WIDTH_B/BYTE_WRITE_WIDTH_B-bit input: Write enable vector
                                       // for port B input data port dinb. 1 bit wide when word-wide writes are
                                       // used. In byte-wide write configurations, each bit controls the
                                       // writing one byte of dinb to address addrb. For example, to
                                       // synchronously write only bits [15-8] of dinb when WRITE_DATA_WIDTH_B
                                       // is 32, web would be 4'b0010.

   );




endmodule
