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

module dma_bypass_controller
(
    // user clk
    input wire          user_clk,
    input wire          user_aresetn,

    // Control Interface
    // axi_lite.slave      s_axil,
    input wire bram_en_a,          // output wire bram_en_a
    input wire bram_we_a,          // output wire [3 : 0] bram_we_a
    input wire[15:0] bram_addr_a,      // output wire [15 : 0] bram_addr_a
    input wire[511:0] bram_wrdata_a,  // output wire [31 : 0] bram_wrdata_a
    output wire[511:0] bram_rddata_a,  // input wire [31 : 0] bram_rddata_a
    // // bypass register
    output reg[31:0][511:0]     bypass_control_reg,
    input wire[31:0][511:0]     bypass_status_reg

);
    reg [63:0][511:0]           bypass_reg;
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

reg[5:0] writeAddr;
reg[5:0] readAddr;

reg[511:0] writeData;
reg[511:0] readData;   

reg[7:0] word_counter;


always @(posedge user_clk)begin
    if (~user_aresetn) begin
        bypass_reg                    <= 0;
    end
    else if(writeState[1])begin
        bypass_reg[writeAddr]         <= writeData;
    end
    else begin
        bypass_reg[31:0]             <= bypass_reg[31:0];
        bypass_reg[63:32]           <= bypass_status_reg;
    end
end

always @(posedge user_clk)begin
    bypass_control_reg                <= bypass_reg[31:0];
end


//handle writes
always @(posedge user_clk)begin
    if (~user_aresetn) begin        
        writeState <= WRITE_IDLE;
    end
    else begin
        case (writeState)
            WRITE_IDLE: begin                
                if (bram_en_a && bram_we_a) begin
                    writeState <= WRITE_DATA;
                    writeAddr <= bram_addr_a[11:6];
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
                if (bram_en_a && (~bram_we_a)) begin
                    readAddr  <= bram_addr_a[11:6];
                    readState <= READ_RESPONSE;
                end
            end
            READ_RESPONSE: begin
                readData <= bypass_reg[readAddr];
                readState <= READ_IDLE;
            end
        endcase
    end
end

endmodule
`default_nettype wire
