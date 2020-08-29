/*
 * Copyright (c) 2019, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 `timescale 1ns / 1ps
`default_nettype none

//`include "davos_types.svh"

module dma_controller
(
    //clk
    input  wire         pcie_clk,
    input  wire         pcie_aresetn,
    // user clk
    input wire          user_clk,
    input wire          user_aresetn,

    // Control Interface
    // axi_lite.slave      s_axil,
    input wire bram_en_a,          // output wire bram_en_a
    input wire[3:0] bram_we_a,          // output wire [3 : 0] bram_we_a
    input wire[11:0] bram_addr_a,      // output wire [15 : 0] bram_addr_a
    input wire[31:0] bram_wrdata_a,  // output wire [31 : 0] bram_wrdata_a
    output wire[31:0] bram_rddata_a,  // input wire [31 : 0] bram_rddata_a
    // // TLB command
    // output reg         m_axis_tlb_interface_valid,
    // input wire         m_axis_tlb_interface_ready,
    // output reg[135:0]  m_axis_tlb_interface_data,

    // //mlweaving parameter
    // output reg         m_axis_mlweaving_valid,
    // input wire         m_axis_mlweaving_ready,
    // output reg[511:0]  m_axis_mlweaving_data,

    // //tlb on same clock
    // input wire[31:0]    tlb_miss_counter,
    // input wire[31:0]    tlb_boundary_crossing_counter,

    // //same clock
    // input wire[31:0]    dma_write_cmd_counter,
    // input wire[31:0]    dma_write_word_counter,
    // input wire[31:0]    dma_write_pkg_counter,
    // input wire[31:0]    dma_read_cmd_counter,
    // input wire[31:0]    dma_read_word_counter,
    // input wire[31:0]    dma_read_pkg_counter,
    // output reg          reset_dma_write_length_counter,
    // input wire[47:0]    dma_write_length_counter,
    // output reg          reset_dma_read_length_counter,
    // input wire[47:0]    dma_read_length_counter,
    // input wire          dma_reads_flushed
    output reg[511:0][31:0]     fpga_control_reg,
    input wire[511:0][31:0]     fpga_status_reg

);
    reg [1023:0][31:0]           fpga_reg;
// localparam AXI_RESP_OK = 2'b00;
// localparam AXI_RESP_SLVERR = 2'b10;


//WRITE states
localparam WRITE_IDLE = 1;
localparam WRITE_DATA = 3;

//READ states
localparam READ_IDLE = 1;
localparam READ_RESPONSE = 3;

// ACTUAL LOGIC

reg[1:0] writeState;
reg[1:0] readState;

reg[9:0] writeAddr;
reg[9:0] readAddr;

reg[31:0] writeData;
reg[31:0] readData;   

reg[7:0] word_counter;


always @(posedge user_clk)begin
    if (~user_aresetn) begin
        fpga_reg                    <= 0;
    end
    else if(writeState[1])begin
        fpga_reg[writeAddr]         <= writeData;
    end
    else begin
        fpga_reg[511:0]             <= fpga_reg[511:0];
        fpga_reg[1023:512]          <= fpga_status_reg;
    end
end

always @(posedge user_clk)begin
    fpga_control_reg                <= fpga_reg[511:0];
end


//handle writes
always @(posedge user_clk)begin
    if (~user_aresetn) begin        
        writeState <= WRITE_IDLE;
    end
    else begin
        case (writeState)
            WRITE_IDLE: begin                
                if (bram_en_a && bram_we_a[0]) begin
                    writeState <= WRITE_DATA;
                    writeAddr <= bram_addr_a[11:2];
                    writeData <= bram_wrdata_a;
                end
            end //WRITE_IDLE
            WRITE_DATA: begin
                writeState <= WRITE_IDLE;  
            end        
        endcase
    end
end

//reads are currently not available
assign bram_rddata_a = readData;
always @(posedge user_clk)
begin
    if (~user_aresetn) begin
        readState <= READ_IDLE;
    end
    else begin      
        case (readState)
            READ_IDLE: begin
                //read_en <= 1;
                if (bram_en_a && (~bram_we_a[0])) begin
                    readAddr  <= bram_addr_a[11:2];
                    readState <= READ_RESPONSE;
                end
            end
            READ_RESPONSE: begin
                readData <= fpga_reg[readAddr];
                readState <= READ_IDLE;
            end
        endcase
    end
end

//ila_axil probe_ila_axil(
//.clk(pcie_clk),

//.probe0(bram_en_a), // input wire [1:0]
//.probe1(bram_we_a), // input wire [4:0]
//.probe2(bram_addr_a), // input wire [16:0]
//.probe3(bram_wrdata_a), // input wire [32:0]
//.probe4(bram_rddata_a), // input wire [32:0]
//.probe5(fpga_reg[0]), // input wire [32:0]
//.probe6(fpga_reg[1]), // input wire [32:0]
//.probe7(fpga_reg[127]) // input wire [32:0]
//);

endmodule
`default_nettype wire
