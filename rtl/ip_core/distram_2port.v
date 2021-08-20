// (C) 2001-2016 Altera Corporation. All rights reserved.
// Your use of Altera Corporation's design tools, logic functions and other 
// software and tools, and its AMPP partner logic functions, and any output 
// files any of the foregoing (including device programming or simulation 
// files), and any associated documentation or information are expressly subject 
// to the terms and conditions of the Altera Program License Subscription 
// Agreement, Altera MegaCore Function License Agreement, or other applicable 
// license agreement, including, without limitation, that your use is for the 
// sole purpose of programming logic devices manufactured by Altera and sold by 
// Altera or its authorized distributors.  Please refer to the applicable 
// agreement for further details.



// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module  distram_2port  #(parameter DATA_WIDTH      = 64,    
                             parameter DEPTH_BIT_WIDTH = 5
) (
    clock,
    data,
    rdaddress,
    wraddress,
    wren,
    q);

    input    clock;
    input  [DATA_WIDTH-1:0]       data;
    input  [DEPTH_BIT_WIDTH-1:0]  rdaddress;
    input  [DEPTH_BIT_WIDTH-1:0]  wraddress;
    input    wren;
    output [DATA_WIDTH-1:0]  q;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_off
`endif
    tri1     clock;
    tri0     wren;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_on
`endif

    wire [DATA_WIDTH-1:0] sub_wire0;
    wire [DATA_WIDTH-1:0] q = sub_wire0[DATA_WIDTH-1:0];

//  <-----Cut code below this line---->

   // xpm_memory_dpdistram: Dual Port Distributed RAM
   // Xilinx Parameterized Macro, version 2019.2

   xpm_memory_dpdistram #(
      .ADDR_WIDTH_A(DEPTH_BIT_WIDTH),               // DECIMAL
      .ADDR_WIDTH_B(DEPTH_BIT_WIDTH),               // DECIMAL
      .BYTE_WRITE_WIDTH_A(DATA_WIDTH),        // DECIMAL
      .CLOCKING_MODE("common_clock"), // String
      .MEMORY_INIT_FILE("none"),      // String
      .MEMORY_INIT_PARAM("0"),        // String
      .MEMORY_OPTIMIZATION("true"),   // String
      .MEMORY_SIZE(2**DEPTH_BIT_WIDTH*DATA_WIDTH),             // DECIMAL
      .MESSAGE_CONTROL(0),            // DECIMAL
      .READ_DATA_WIDTH_A(DATA_WIDTH),         // DECIMAL
      .READ_DATA_WIDTH_B(DATA_WIDTH),         // DECIMAL
      .READ_LATENCY_A(2),             // DECIMAL
      .READ_LATENCY_B(2),             // DECIMAL
      .READ_RESET_VALUE_A("0"),       // String
      .READ_RESET_VALUE_B("0"),       // String
      .RST_MODE_A("SYNC"),            // String
      .RST_MODE_B("SYNC"),            // String
      .SIM_ASSERT_CHK(0),             // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
      .USE_EMBEDDED_CONSTRAINT(0),    // DECIMAL
      .USE_MEM_INIT(1),               // DECIMAL
      .WRITE_DATA_WIDTH_A(DATA_WIDTH)         // DECIMAL
   )
   xpm_memory_dpdistram_inst (
      .douta(),   // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
      .doutb(sub_wire0),   // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
      .addra(wraddress),   // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
      .addrb(rdaddress),   // ADDR_WIDTH_B-bit input: Address for port B write and read operations.
      .clka(clock),     // 1-bit input: Clock signal for port A. Also clocks port B when parameter CLOCKING_MODE
                       // is "common_clock".

      .clkb(clock),     // 1-bit input: Clock signal for port B when parameter CLOCKING_MODE is
                       // "independent_clock". Unused when parameter CLOCKING_MODE is "common_clock".

      .dina(data),     // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
      .ena(1'b1),       // 1-bit input: Memory enable signal for port A. Must be high on clock cycles when read
                       // or write operations are initiated. Pipelined internally.

      .enb(1'b1),       // 1-bit input: Memory enable signal for port B. Must be high on clock cycles when read
                       // or write operations are initiated. Pipelined internally.

      .regcea(1'b1), // 1-bit input: Clock Enable for the last register stage on the output data path.
      .regceb(1'b1), // 1-bit input: Do not change from the provided value.
      .rsta(1'b0),     // 1-bit input: Reset signal for the final port A output register stage. Synchronously
                       // resets output port douta to the value specified by parameter READ_RESET_VALUE_A.

      .rstb(1'b0),     // 1-bit input: Reset signal for the final port B output register stage. Synchronously
                       // resets output port doutb to the value specified by parameter READ_RESET_VALUE_B.

      .wea(wren)        // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector for port A input
                       // data port dina. 1 bit wide when word-wide writes are used. In byte-wide write
                       // configurations, each bit controls the writing one byte of dina to address addra. For
                       // example, to synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A is
                       // 32, wea would be 4'b0010.

   );

   // End of xpm_memory_dpdistram_inst instantiation
				



endmodule


