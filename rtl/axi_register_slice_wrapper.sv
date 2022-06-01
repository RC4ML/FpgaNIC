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

module axi_register_slice_wrapper #(
    parameter WIDTH = 512
) (
    input wire          aclk,
    input wire          aresetn,
    axi_mm.slave        s_axi,
    axi_mm.master       m_axi
);



        axi_register_slice_512 axi_register_slice_512_inst (
            .aclk(aclk),                      // input wire aclk
            .aresetn(aresetn),                // input wire aresetn
            .s_axi_awaddr   (s_axi.awaddr),      // input wire [33 : 0] s_axi_awaddr
            .s_axi_awlen    (s_axi.awlen),        // input wire [7 : 0] s_axi_awlen
            .s_axi_awsize   (s_axi.awsize),      // input wire [2 : 0] s_axi_awsize
            .s_axi_awburst  (s_axi.awburst),    // input wire [1 : 0] s_axi_awburst
            .s_axi_awlock   (s_axi.awlock),      // input wire [0 : 0] s_axi_awlock
            .s_axi_awcache  (s_axi.awcache),    // input wire [3 : 0] s_axi_awcache
            .s_axi_awprot   (s_axi.awprot),      // input wire [2 : 0] s_axi_awprot
            .s_axi_awregion (s_axi.awregion),  // input wire [3 : 0] s_axi_awregion
            .s_axi_awqos    (s_axi.awqos),        // input wire [3 : 0] s_axi_awqos
            .s_axi_awvalid  (s_axi.awvalid),    // input wire s_axi_awvalid
            .s_axi_awready  (s_axi.awready),    // output wire s_axi_awready
            .s_axi_wdata    (s_axi.wdata),        // input wire [511 : 0] s_axi_wdata
            .s_axi_wstrb    (s_axi.wstrb),        // input wire [63 : 0] s_axi_wstrb
            .s_axi_wlast    (s_axi.wlast),        // input wire s_axi_wlast
            .s_axi_wvalid   (s_axi.wvalid),      // input wire s_axi_wvalid
            .s_axi_wready   (s_axi.wready),      // output wire s_axi_wready
            .s_axi_bresp    (s_axi.bresp),        // output wire [1 : 0] s_axi_bresp
            .s_axi_bvalid   (s_axi.bvalid),      // output wire s_axi_bvalid
            .s_axi_bready   (s_axi.bready),      // input wire s_axi_bready
            .s_axi_araddr   (s_axi.araddr),      // input wire [33 : 0] s_axi_araddr
            .s_axi_arlen    (s_axi.arlen),        // input wire [7 : 0] s_axi_arlen
            .s_axi_arsize   (s_axi.arsize),      // input wire [2 : 0] s_axi_arsize
            .s_axi_arburst  (s_axi.arburst),    // input wire [1 : 0] s_axi_arburst
            .s_axi_arlock   (s_axi.arlock),      // input wire [0 : 0] s_axi_arlock
            .s_axi_arcache  (s_axi.arcache),    // input wire [3 : 0] s_axi_arcache
            .s_axi_arprot   (s_axi.arprot),      // input wire [2 : 0] s_axi_arprot
            .s_axi_arregion (s_axi.arregion),  // input wire [3 : 0] s_axi_arregion
            .s_axi_arqos    (s_axi.arqos),        // input wire [3 : 0] s_axi_arqos
            .s_axi_arvalid  (s_axi.arvalid),    // input wire s_axi_arvalid
            .s_axi_arready  (s_axi.arready),    // output wire s_axi_arready
            .s_axi_rdata    (s_axi.rdata),        // output wire [511 : 0] s_axi_rdata
            .s_axi_rresp    (s_axi.rresp),        // output wire [1 : 0] s_axi_rresp
            .s_axi_rlast    (s_axi.rlast),        // output wire s_axi_rlast
            .s_axi_rvalid   (s_axi.rvalid),      // output wire s_axi_rvalid
            .s_axi_rready   (s_axi.rready),      // input wire s_axi_rready
            .m_axi_awaddr   (m_axi.awaddr),      // output wire [33 : 0] m_axi_awaddr
            .m_axi_awlen    (m_axi.awlen),        // output wire [7 : 0] m_axi_awlen
            .m_axi_awsize   (m_axi.awsize),      // output wire [2 : 0] m_axi_awsize
            .m_axi_awburst  (m_axi.awburst),    // output wire [1 : 0] m_axi_awburst
            .m_axi_awlock   (m_axi.awlock),      // output wire [0 : 0] m_axi_awlock
            .m_axi_awcache  (m_axi.awcache),    // output wire [3 : 0] m_axi_awcache
            .m_axi_awprot   (m_axi.awprot),      // output wire [2 : 0] m_axi_awprot
            .m_axi_awregion (m_axi.awregion),  // output wire [3 : 0] m_axi_awregion
            .m_axi_awqos    (m_axi.awqos),        // output wire [3 : 0] m_axi_awqos
            .m_axi_awvalid  (m_axi.awvalid),    // output wire m_axi_awvalid
            .m_axi_awready  (m_axi.awready),    // input wire m_axi_awready
            .m_axi_wdata    (m_axi.wdata),        // output wire [511 : 0] m_axi_wdata
            .m_axi_wstrb    (m_axi.wstrb),        // output wire [63 : 0] m_axi_wstrb
            .m_axi_wlast    (m_axi.wlast),        // output wire m_axi_wlast
            .m_axi_wvalid   (m_axi.wvalid),      // output wire m_axi_wvalid
            .m_axi_wready   (m_axi.wready),      // input wire m_axi_wready
            .m_axi_bresp    (m_axi.bresp),        // input wire [1 : 0] m_axi_bresp
            .m_axi_bvalid   (m_axi.bvalid),      // input wire m_axi_bvalid
            .m_axi_bready   (m_axi.bready),      // output wire m_axi_bready
            .m_axi_araddr   (m_axi.araddr),      // output wire [33 : 0] m_axi_araddr
            .m_axi_arlen    (m_axi.arlen),        // output wire [7 : 0] m_axi_arlen
            .m_axi_arsize   (m_axi.arsize),      // output wire [2 : 0] m_axi_arsize
            .m_axi_arburst  (m_axi.arburst),    // output wire [1 : 0] m_axi_arburst
            .m_axi_arlock   (m_axi.arlock),      // output wire [0 : 0] m_axi_arlock
            .m_axi_arcache  (m_axi.arcache),    // output wire [3 : 0] m_axi_arcache
            .m_axi_arprot   (m_axi.arprot),      // output wire [2 : 0] m_axi_arprot
            .m_axi_arregion (m_axi.arregion),  // output wire [3 : 0] m_axi_arregion
            .m_axi_arqos    (m_axi.arqos),        // output wire [3 : 0] m_axi_arqos
            .m_axi_arvalid  (m_axi.arvalid),    // output wire m_axi_arvalid
            .m_axi_arready  (m_axi.arready),    // input wire m_axi_arready
            .m_axi_rdata    (m_axi.rdata),        // input wire [511 : 0] m_axi_rdata
            .m_axi_rresp    (m_axi.rresp),        // input wire [1 : 0] m_axi_rresp
            .m_axi_rlast    (m_axi.rlast),        // input wire m_axi_rlast
            .m_axi_rvalid   (m_axi.rvalid),      // input wire m_axi_rvalid
            .m_axi_rready   (m_axi.rready)      // output wire m_axi_rready
          );      




endmodule
`default_nettype wire