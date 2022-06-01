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
    input wire          pcie_clk,
    input wire          pcie_aresetn,    
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
    axis_meta.master    axis_tcp_recv_read_cnt,
    axis_meta.master    bypass_cmd,

    ////off path cmd
    axis_meta.master    m_axis_get_data_cmd,
    axis_meta.master    m_axis_put_data_cmd,
    //one side
    axis_meta.master    m_axis_get_data_form_net,
    axis_meta.master    m_axis_put_data_to_net   

);
// localparam AXI_RESP_OK = 2'b00;
// localparam AXI_RESP_SLVERR = 2'b10;


//WRITE states
localparam WRITE_IDLE = 0;
localparam SEND_WRITE = 1;
localparam RECV_READ  = 2;
localparam WRITE_DATA = 3;
localparam GET_DATA = 4;
localparam PUT_DATA = 5;
localparam GET_DATA_FROM_NET = 6;
localparam PUT_DATA_TO_NET = 7;

// ACTUAL LOGIC
axi_stream s_bypass_cmd();

reg[3:0] writeState;
reg[5:0] writeAddr;
reg[511:0] writeData;   


// axis_data_fifo_512_cc bypass_cmd_fifo (
//   .s_axis_aresetn(pcie_aresetn),  // input wire s_axis_aresetn
//   .s_axis_aclk(pcie_clk),        // input wire s_axis_aclk
//   .s_axis_tvalid(s_bypass_cmd.valid),    // input wire s_axis_tvalid
//   .s_axis_tready(s_bypass_cmd.ready),    // output wire s_axis_tready
//   .s_axis_tdata(s_bypass_cmd.data),      // input wire [511 : 0] s_axis_tdata
//   .s_axis_tkeep(s_bypass_cmd.keep),      // input wire [63 : 0] s_axis_tkeep
//   .s_axis_tlast(s_bypass_cmd.last),      // input wire s_axis_tlast
//   .m_axis_aclk(user_clk),        // input wire m_axis_aclk
//   .m_axis_tvalid(bypass_cmd.valid),    // output wire m_axis_tvalid
//   .m_axis_tready(bypass_cmd.ready),    // input wire m_axis_tready
//   .m_axis_tdata(bypass_cmd.data),      // output wire [511 : 0] m_axis_tdata
//   .m_axis_tkeep(bypass_cmd.keep),      // output wire [63 : 0] m_axis_tkeep
//   .m_axis_tlast(bypass_cmd.last)      // output wire m_axis_tlast
// );

assign bypass_cmd.valid   = writeState ==  WRITE_DATA;
assign bypass_cmd.data    = {writeData[111:96],writeData[79:0]};

// assign axis_tcp_send_write_cnt.valid   = writeState ==  SEND_WRITE;
// assign axis_tcp_send_write_cnt.data    = writeData;

assign axis_tcp_recv_read_cnt.valid   = writeState ==  RECV_READ;
assign axis_tcp_recv_read_cnt.data    = writeData[79:0];

///////off path cmd

assign m_axis_get_data_cmd.valid   = writeState ==  GET_DATA;
assign m_axis_get_data_cmd.data    = writeData[159:0];

assign m_axis_put_data_cmd.valid   = writeState ==  PUT_DATA;
assign m_axis_put_data_cmd.data    = writeData[159:0];

////////////one side

assign m_axis_get_data_form_net.valid   = writeState ==  GET_DATA_FROM_NET;
assign m_axis_get_data_form_net.data    = writeData[127:0];

assign m_axis_put_data_to_net.valid   = writeState ==  PUT_DATA_TO_NET;
assign m_axis_put_data_to_net.data    = writeData[127:0];


//handle writes
always @(posedge pcie_clk)begin
    if (~pcie_aresetn) begin        
        writeState <= WRITE_IDLE;
    end
    else begin
        case (writeState)
            WRITE_IDLE: begin                
                if (bram_en_a && bram_we_a) begin
                    // if(bram_addr_a[11:6] == 1)begin
                    //     writeState <= SEND_WRITE;
                    //     writeAddr <= bram_addr_a[11:6];
                    //     writeData <= bram_wrdata_a;
                    // end
                    if(bram_addr_a[11:6] == 2)begin
                        writeState <= RECV_READ;
                        writeAddr <= bram_addr_a[11:6];
                        writeData <= bram_wrdata_a;
                    end
                    else if(bram_addr_a[11:6] == 3)begin
                        writeState <= WRITE_DATA;
                        writeAddr <= bram_addr_a[11:6];
                        writeData <= bram_wrdata_a;
                    end   
                    else if(bram_addr_a[11:6] == 4)begin
                        writeState <= GET_DATA;
                        writeAddr <= bram_addr_a[11:6];
                        writeData <= bram_wrdata_a;
                    end 
                    else if(bram_addr_a[11:6] == 5)begin
                        writeState <= PUT_DATA;
                        writeAddr <= bram_addr_a[11:6];
                        writeData <= bram_wrdata_a;
                    end  
                    else if(bram_addr_a[11:6] == 6)begin
                        writeState <= GET_DATA_FROM_NET;
                        writeAddr <= bram_addr_a[11:6];
                        writeData <= bram_wrdata_a;
                    end  
                    else if(bram_addr_a[11:6] == 7)begin
                        writeState <= PUT_DATA_TO_NET;
                        writeAddr <= bram_addr_a[11:6];
                        writeData <= bram_wrdata_a;
                    end                                                                                                  
                    else begin
                        writeState <= WRITE_IDLE;
                    end
                end
                else begin
                    writeState <= WRITE_IDLE;
                end
            end //WRITE_IDLE
            // SEND_WRITE: begin
            //     if(axis_tcp_send_write_cnt.valid & axis_tcp_send_write_cnt.ready)begin
            //         writeState <= WRITE_IDLE; 
            //     end
            //     else begin
            //         writeState <= SEND_WRITE;
            //     end 
            // end 
            RECV_READ: begin
                if(axis_tcp_recv_read_cnt.valid & axis_tcp_recv_read_cnt.ready)begin
                    writeState <= WRITE_IDLE; 
                end
                else begin
                    writeState <= RECV_READ;
                end 
            end                         
            WRITE_DATA: begin
                if(bypass_cmd.valid & bypass_cmd.ready)begin
                    writeState <= WRITE_IDLE; 
                end
                else begin
                    writeState <= WRITE_DATA;
                end 
            end 
            GET_DATA: begin
                if(m_axis_get_data_cmd.valid & m_axis_get_data_cmd.ready)begin
                    writeState <= WRITE_IDLE; 
                end
                else begin
                    writeState <= GET_DATA;
                end 
            end 
            PUT_DATA: begin
                if(m_axis_put_data_cmd.valid & m_axis_put_data_cmd.ready)begin
                    writeState <= WRITE_IDLE; 
                end
                else begin
                    writeState <= PUT_DATA;
                end 
            end
            GET_DATA_FROM_NET: begin
                if(m_axis_get_data_form_net.valid & m_axis_get_data_form_net.ready)begin
                    writeState <= WRITE_IDLE; 
                end
                else begin
                    writeState <= GET_DATA_FROM_NET;
                end 
            end 
            PUT_DATA_TO_NET: begin
                if(m_axis_put_data_to_net.valid & m_axis_put_data_to_net.ready)begin
                    writeState <= WRITE_IDLE; 
                end
                else begin
                    writeState <= PUT_DATA_TO_NET;
                end 
            end                                             
        endcase
    end
end

//ila_1 bypass (
//	.clk(pcie_clk), // input wire clk


//	.probe0(bram_en_a), // input wire [0:0]  probe0  
//	.probe1(bram_we_a), // input wire [0:0]  probe1 
//	.probe2(bram_addr_a), // input wire [15:0]  probe2 
//	.probe3(bram_wrdata_a), // input wire [511:0]  probe3 
//	.probe4(s_bypass_cmd.valid), // input wire [0:0]  probe4 
//	.probe5(s_bypass_cmd.ready), // input wire [0:0]  probe5 
//	.probe6(bypass_cmd.valid), // input wire [0:0]  probe6 
//	.probe7(bypass_cmd.ready) // input wire [0:0]  probe7
//);


endmodule
`default_nettype wire
