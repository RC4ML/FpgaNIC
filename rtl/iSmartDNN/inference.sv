/*
 * Copyright (c) 2020, Systems Group, ETH Zurich
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
 
 `include "example_module.vh"
 
 module inference (
    input wire         clk,
    input wire         rstn,

    //DMA Commands
    axis_mem_cmd.master         axis_dma_read_cmd,
    //DMA Data streams      
	axi_stream.slave            axis_dma_read_data,    
    
     
	//tcp send
    axis_meta.master     		m_axis_tx_metadata,
    axi_stream.master    		m_axis_tx_data,
    axis_meta.slave    			s_axis_tx_status,

	//tcp recv    
    axis_meta.slave    			s_axis_rx_metadata,
    axi_stream.slave   			s_axis_rx_data,

    // control reg
    input wire[15:0][31:0]      control_reg,
    output wire[15:0][31:0]      status_reg     
 
 );

axis_meta #(.WIDTH(88))     axis_tcp_rx_meta0();
axi_stream #(.WIDTH(512))   axis_tcp_rx_data0();

axis_meta #(.WIDTH(88))     axis_tcp_rx_meta1();
axi_stream #(.WIDTH(512))   axis_tcp_rx_data1();

axis_meta #(.WIDTH(48))     axis_tcp_tx_meta0();
axi_stream #(.WIDTH(512))   axis_tcp_tx_data0(); 

axis_meta #(.WIDTH(48))     axis_tcp_tx_meta1();
axi_stream #(.WIDTH(512))   axis_tcp_tx_data1(); 

    reg                         choice;

    always@(posedge clk)begin
        if(control_reg[0][0])begin
            choice              <= 1;
        end
        else begin
            choice              <= 0;
        end
    end


    assign s_axis_tx_status.ready = 1;




    assign s_axis_rx_metadata.ready     = choice ? axis_tcp_rx_meta1.ready : axis_tcp_rx_meta0.ready;

    assign axis_tcp_rx_meta0.valid      = choice ? 0 : s_axis_rx_metadata.valid;
    assign axis_tcp_rx_meta0.data       = choice ? 0 : s_axis_rx_metadata.data;
    assign axis_tcp_rx_meta1.valid      = choice ? s_axis_rx_metadata.valid : 0;
    assign axis_tcp_rx_meta1.data       = choice ? s_axis_rx_metadata.data : 0;

    assign s_axis_rx_data.ready         = choice ? axis_tcp_rx_data1.ready : axis_tcp_rx_data0.ready;

    assign axis_tcp_rx_data0.valid      = choice ? 0 : s_axis_rx_data.valid;
    assign axis_tcp_rx_data0.data       = choice ? 0 : s_axis_rx_data.data;
    assign axis_tcp_rx_data0.keep       = choice ? 0 : s_axis_rx_data.keep;
    assign axis_tcp_rx_data0.last       = choice ? 0 : s_axis_rx_data.last;

    assign axis_tcp_rx_data1.valid      = choice ? s_axis_rx_data.valid : 0;
    assign axis_tcp_rx_data1.data       = choice ? s_axis_rx_data.data : 0;
    assign axis_tcp_rx_data1.keep       = choice ? s_axis_rx_data.keep : 0;
    assign axis_tcp_rx_data1.last       = choice ? s_axis_rx_data.last : 0;


    assign m_axis_tx_metadata.valid     = choice ? axis_tcp_tx_meta1.valid : axis_tcp_tx_meta0.valid;
    assign m_axis_tx_metadata.data      = choice ? axis_tcp_tx_meta1.data : axis_tcp_tx_meta0.data;

    assign axis_tcp_tx_meta0.ready      = choice ? 0 : m_axis_tx_metadata.ready;
    assign axis_tcp_tx_meta1.ready      = choice ? m_axis_tx_metadata.ready : 0;

    assign m_axis_tx_data.valid         = choice ? axis_tcp_tx_data1.valid : axis_tcp_tx_data0.valid;
    assign m_axis_tx_data.data          = choice ? axis_tcp_tx_data1.data : axis_tcp_tx_data0.data;
    assign m_axis_tx_data.keep          = choice ? axis_tcp_tx_data1.keep : axis_tcp_tx_data0.keep;
    assign m_axis_tx_data.last          = choice ? axis_tcp_tx_data1.last : axis_tcp_tx_data0.last;

    assign axis_tcp_tx_data0.ready      = choice ? 0 : m_axis_tx_data.ready;
    assign axis_tcp_tx_data1.ready      = choice ? m_axis_tx_data.ready : 0;


mobilenet mobilenet_inst (
    .clk(pcie_clk),
    .rstn(user_rstn),
   
    
    //tcp app interface streams
    .m_axis_tx_metadata(m_axis_tx_metadata),
    .m_axis_tx_data(m_axis_tx_data),  

    .s_axis_rx_metadata(s_axis_rx_metadata),
    .s_axis_rx_data(s_axis_rx_data),


   //control reg
    // .control_reg                    (control_reg),
    .status_reg                     (status_reg[0])     

);






send_inference send_inference_inst (
    .clk(pcie_clk),
    .rstn(user_rstn),

    .m_axis_dma_read_cmd(axis_dma_read_cmd),
    .s_axis_dma_read_data(axis_dma_read_data),   
    
    //tcp app interface streams
    .m_axis_tx_metadata(m_axis_tx_metadata),
    .m_axis_tx_data(m_axis_tx_data),  

    .s_axis_rx_metadata(s_axis_rx_metadata),
    .s_axis_rx_data(s_axis_rx_data),


   //control reg
    .control_reg                    (control_reg[15:1]),
    .status_reg                     (status_reg[8:1])     

);


 endmodule
 `default_nettype wire
 