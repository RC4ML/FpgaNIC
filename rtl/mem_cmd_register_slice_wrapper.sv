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

module mem_cmd_register_slice_wrapper #(
    parameter DIRECT = 0
) (
    input wire          aclk,
    input wire          aresetn,
    axis_mem_cmd.slave    s_axis,
    axis_mem_cmd.master   m_axis
);



generate
    if(DIRECT == 0)begin
        axis_register_slice_96 slice_inst (
            .aclk(aclk),                    // input wire aclk
            .aresetn(aresetn),              // input wire aresetn
            .s_axis_tvalid(s_axis.valid),  // input wire s_axis_tvalid
            .s_axis_tready(s_axis.ready),  // output wire s_axis_tready
            .s_axis_tdata({s_axis.address,s_axis.length}),    // input wire [95 : 0] s_axis_tdata
            .m_axis_tvalid(m_axis.valid),  // output wire m_axis_tvalid
            .m_axis_tready(m_axis.ready),  // input wire m_axis_tready
            .m_axis_tdata({m_axis.address,m_axis.length})    // output wire [95 : 0] m_axis_tdata
        );      
    end
    else begin
        assign m_axis.valid     = s_axis.valid;
        assign m_axis.address   = s_axis.address;
        assign m_axis.length    = s_axis.length;
        assign s_axis.ready     = m_axis.ready;
    end
endgenerate




endmodule
`default_nettype wire