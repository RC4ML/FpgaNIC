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
 
 module mobilenet (
     input wire         clk,
     input wire         rstn,
    
     
	//tcp send
    axis_meta.master     		m_axis_tx_metadata,
    axi_stream.master    		m_axis_tx_data,

	//tcp recv    
    axis_meta.slave    			s_axis_rx_metadata,
    axi_stream.slave   			s_axis_rx_data,

    // control reg
    //  input wire[15:0][31:0]      control_reg,
     output wire[31:0]      status_reg     
 
 );

 axi_mm #(.DATA_WIDTH(32),.ADDR_WIDTH(32))         m_axi_img();
 axi_mm #(.DATA_WIDTH(32),.ADDR_WIDTH(32))         m_axi_input();






reg 							s_axi_awvalid;
wire 							s_axi_awready;
reg[31:0]						s_axi_awaddr;
wire 							s_axi_wlast;	
reg[31:0]						length,num_mem_ops_minus_1;
reg             [31:0]              wr_ops;
reg             [7 :0]              burst_inc;
reg                                 wr_data_done; 
reg[15:0]						session_id;


always@(posedge clk)begin
	if(~rstn)begin
		session_id						<= 0;
	end
	else if(s_axis_rx_metadata.valid & s_axis_rx_metadata.ready)begin
		session_id						<= s_axis_rx_metadata[15:0];
	end
	else begin
		session_id						<= session_id;
	end
end




always @(posedge clk)begin
	if (~rstn)begin
		s_axi_awaddr                    <= 8'b0;
	end
	else if (s_axi_awvalid & s_axi_awready)begin
		s_axi_awaddr                    <= s_axi_awaddr + 512;
		if (s_axi_awaddr == (length - 512)) begin
			s_axi_awaddr                <= 8'b0;
		end
	end
	else begin
		s_axi_awaddr                    <= s_axi_awaddr;
		
	end
end

always @(posedge clk)begin
	if (~rstn)begin
		s_axi_awvalid                 <= 1'b0;
	end
	else if((s_axi_awaddr == (length - 512)) && s_axi_awvalid & s_axi_awready) begin
		s_axi_awvalid                 <= 1'b0;
	end 
	else if (s_axis_rx_metadata.ready & s_axis_rx_metadata.valid)begin
		s_axi_awvalid                 <= 1'b1;
	end
	else begin
		s_axi_awvalid                 <= s_axi_awvalid;
	end
end

always @(posedge clk)begin
	if (~rstn)begin
		burst_inc                   <= 8'b0;
		wr_ops                      <= 32'b0;
		wr_data_done                <= 1'b0;            
	end       
	else if (s_axis_rx_data.valid & s_axis_rx_data.ready)begin
		burst_inc                   <= burst_inc + 8'b1;
		if (burst_inc == 4'h7) begin
			burst_inc               <= 8'b0;
			wr_ops                  <= wr_ops + 1'b1;
			if (wr_ops == num_mem_ops_minus_1)begin
				wr_data_done        <= 1'b1;
			end                
		end
	end
end


always@(posedge clk)begin
	length							<= 32'h26400;
	num_mem_ops_minus_1             <= (s_axi_awaddr>>>9) -1;
end

assign s_axis_rx_metadata.ready = 1;
assign s_axi_awvalid = 1;
assign s_axi_wlast = (burst_inc == 4'h7)&s_axis_rx_data.valid & s_axis_rx_data.ready;

axi_dwidth_converter_512_32 axi_dwidth_converter_512_32 (
  .s_axi_aclk(clk),          // input wire s_axi_aclk
  .s_axi_aresetn(rstn),    // input wire s_axi_aresetn
  .s_axi_awaddr		(s_axi_awaddr),      // input wire [31 : 0] s_axi_awaddr
  .s_axi_awlen		(4'h7),        // input wire [7 : 0] s_axi_awlen
  .s_axi_awsize		(3'b110),      // input wire [2 : 0] s_axi_awsize
  .s_axi_awburst	(2'b01),    // input wire [1 : 0] s_axi_awburst
  .s_axi_awlock		(0),      // input wire [0 : 0] s_axi_awlock
  .s_axi_awcache	(0),    // input wire [3 : 0] s_axi_awcache
  .s_axi_awprot		(0),      // input wire [2 : 0] s_axi_awprot
  .s_axi_awregion	(0),  // input wire [3 : 0] s_axi_awregion
  .s_axi_awqos		(0),        // input wire [3 : 0] s_axi_awqos
  .s_axi_awvalid	(s_axi_awvalid),    // input wire s_axi_awvalid
  .s_axi_awready	(s_axi_awready),    // output wire s_axi_awready
  .s_axi_wdata		(s_axis_rx_data.data),        // input wire [511 : 0] s_axi_wdata
  .s_axi_wstrb		(64'hffff_ffff_ffff_ffff),        // input wire [63 : 0] s_axi_wstrb
  .s_axi_wlast		(s_axi_wlast),        // input wire s_axi_wlast
  .s_axi_wvalid		(s_axis_rx_data.valid),      // input wire s_axi_wvalid
  .s_axi_wready		(s_axis_rx_data.ready),      // output wire s_axi_wready
  .s_axi_bresp		(2'h3),        // output wire [1 : 0] s_axi_bresp
  .s_axi_bvalid		(),      // output wire s_axi_bvalid
  .s_axi_bready		(1),      // input wire s_axi_bready
  .s_axi_araddr		(0),      // input wire [31 : 0] s_axi_araddr
  .s_axi_arlen		(0),        // input wire [7 : 0] s_axi_arlen
  .s_axi_arsize		(0),      // input wire [2 : 0] s_axi_arsize
  .s_axi_arburst	(0),    // input wire [1 : 0] s_axi_arburst
  .s_axi_arlock		(0),      // input wire [0 : 0] s_axi_arlock
  .s_axi_arcache	(0),    // input wire [3 : 0] s_axi_arcache
  .s_axi_arprot		(0),      // input wire [2 : 0] s_axi_arprot
  .s_axi_arregion	(0),  // input wire [3 : 0] s_axi_arregion
  .s_axi_arqos		(0),        // input wire [3 : 0] s_axi_arqos
  .s_axi_arvalid	(0),    // input wire s_axi_arvalid
  .s_axi_arready	(),    // output wire s_axi_arready
  .s_axi_rdata		(),        // output wire [511 : 0] s_axi_rdata
  .s_axi_rresp		(),        // output wire [1 : 0] s_axi_rresp
  .s_axi_rlast		(),        // output wire s_axi_rlast
  .s_axi_rvalid		(),      // output wire s_axi_rvalid
  .s_axi_rready		(0),      // input wire s_axi_rready
  .m_axi_awaddr		(m_axi_img.awaddr),      // output wire [31 : 0] m_axi_awaddr
  .m_axi_awlen		(m_axi_img.awlen),        // output wire [7 : 0] m_axi_awlen
  .m_axi_awsize		(m_axi_img.awsize),      // output wire [2 : 0] m_axi_awsize
  .m_axi_awburst	(m_axi_img.awburst),    // output wire [1 : 0] m_axi_awburst
  .m_axi_awlock		(m_axi_img.awlock),      // output wire [0 : 0] m_axi_awlock
  .m_axi_awcache	(m_axi_img.awcache),    // output wire [3 : 0] m_axi_awcache
  .m_axi_awprot		(m_axi_img.awprot),      // output wire [2 : 0] m_axi_awprot
  .m_axi_awregion	(m_axi_img.awregion),  // output wire [3 : 0] m_axi_awregion
  .m_axi_awqos		(m_axi_img.awqos),        // output wire [3 : 0] m_axi_awqos
  .m_axi_awvalid	(m_axi_img.awvalid),    // output wire m_axi_awvalid
  .m_axi_awready	(m_axi_img.awready),    // input wire m_axi_awready
  .m_axi_wdata		(m_axi_img.wdata),        // output wire [31 : 0] m_axi_wdata
  .m_axi_wstrb		(m_axi_img.wstrb),        // output wire [3 : 0] m_axi_wstrb
  .m_axi_wlast		(m_axi_img.wlast),        // output wire m_axi_wlast
  .m_axi_wvalid		(m_axi_img.wvalid),      // output wire m_axi_wvalid
  .m_axi_wready		(m_axi_img.wready),      // input wire m_axi_wready
  .m_axi_bresp		(m_axi_img.bresp),        // input wire [1 : 0] m_axi_bresp
  .m_axi_bvalid		(m_axi_img.bvalid),      // input wire m_axi_bvalid
  .m_axi_bready		(m_axi_img.bready),      // output wire m_axi_bready
  .m_axi_araddr		(),      // output wire [31 : 0] m_axi_araddr
  .m_axi_arlen		(),        // output wire [7 : 0] m_axi_arlen
  .m_axi_arsize		(),      // output wire [2 : 0] m_axi_arsize
  .m_axi_arburst	(),    // output wire [1 : 0] m_axi_arburst
  .m_axi_arlock		(),      // output wire [0 : 0] m_axi_arlock
  .m_axi_arcache	(),    // output wire [3 : 0] m_axi_arcache
  .m_axi_arprot		(),      // output wire [2 : 0] m_axi_arprot
  .m_axi_arregion	(),  // output wire [3 : 0] m_axi_arregion
  .m_axi_arqos		(),        // output wire [3 : 0] m_axi_arqos
  .m_axi_arvalid	(),    // output wire m_axi_arvalid
  .m_axi_arready	(0),    // input wire m_axi_arready
  .m_axi_rdata		(0),        // input wire [31 : 0] m_axi_rdata
  .m_axi_rresp		(0),        // input wire [1 : 0] m_axi_rresp
  .m_axi_rlast		(0),        // input wire m_axi_rlast
  .m_axi_rvalid		(0),      // input wire m_axi_rvalid
  .m_axi_rready		()      // output wire m_axi_rready
);




infer_ram img_ram (
  .s_aclk       (clk),                // input wire s_aclk
  .s_aresetn    (rstn),          // input wire s_aresetn
  .s_axi_awid   (m_axi_img.awid),        // input wire [3 : 0] s_axi_awid
  .s_axi_awaddr (m_axi_img.awaddr),    // input wire [31 : 0] s_axi_awaddr
  .s_axi_awlen  (m_axi_img.awlen),      // input wire [7 : 0] s_axi_awlen
  .s_axi_awsize (m_axi_img.awsize),    // input wire [2 : 0] s_axi_awsize
  .s_axi_awburst(m_axi_img.awburst),  // input wire [1 : 0] s_axi_awburst
  .s_axi_awvalid(m_axi_img.awvalid),  // input wire s_axi_awvalid
  .s_axi_awready(m_axi_img.awready),  // output wire s_axi_awready
  .s_axi_wdata  (m_axi_img.wdata),      // input wire [31 : 0] s_axi_wdata
  .s_axi_wstrb  (m_axi_img.wstrb),      // input wire [3 : 0] s_axi_wstrb
  .s_axi_wlast  (m_axi_img.wlast),      // input wire s_axi_wlast
  .s_axi_wvalid (m_axi_img.wvalid),    // input wire s_axi_wvalid
  .s_axi_wready (m_axi_img.wready),    // output wire s_axi_wready
  .s_axi_bid    (m_axi_img.bid),          // output wire [3 : 0] s_axi_bid
  .s_axi_bresp  (m_axi_img.bresp),      // output wire [1 : 0] s_axi_bresp
  .s_axi_bvalid (m_axi_img.bvalid),    // output wire s_axi_bvalid
  .s_axi_bready (m_axi_img.bready),    // input wire s_axi_bready
  .s_axi_arid   (m_axi_img.arid),        // input wire [3 : 0] s_axi_arid
  .s_axi_araddr (m_axi_img.araddr),    // input wire [31 : 0] s_axi_araddr
  .s_axi_arlen  (m_axi_img.arlen),      // input wire [7 : 0] s_axi_arlen
  .s_axi_arsize (m_axi_img.arsize),    // input wire [2 : 0] s_axi_arsize
  .s_axi_arburst(m_axi_img.arburst),  // input wire [1 : 0] s_axi_arburst
  .s_axi_arvalid(m_axi_img.arvalid),  // input wire s_axi_arvalid
  .s_axi_arready(m_axi_img.arready),  // output wire s_axi_arready
  .s_axi_rid    (m_axi_img.rid),          // output wire [3 : 0] s_axi_rid
  .s_axi_rdata  (m_axi_img.rdata),      // output wire [31 : 0] s_axi_rdata
  .s_axi_rresp  (m_axi_img.rresp),      // output wire [1 : 0] s_axi_rresp
  .s_axi_rlast  (m_axi_img.rlast),      // output wire s_axi_rlast
  .s_axi_rvalid (m_axi_img.rvalid),    // output wire s_axi_rvalid
  .s_axi_rready (m_axi_img.rready)    // input wire s_axi_rready
);


infer_ram input_ram (
  .s_aclk       (clk),                // input wire s_aclk
  .s_aresetn    (rstn),          // input wire s_aresetn
  .s_axi_awid   (m_axi_input.awid),        // input wire [3 : 0] s_axi_awid
  .s_axi_awaddr (m_axi_input.awaddr),    // input wire [31 : 0] s_axi_awaddr
  .s_axi_awlen  (m_axi_input.awlen),      // input wire [7 : 0] s_axi_awlen
  .s_axi_awsize (m_axi_input.awsize),    // input wire [2 : 0] s_axi_awsize
  .s_axi_awburst(m_axi_input.awburst),  // input wire [1 : 0] s_axi_awburst
  .s_axi_awvalid(m_axi_input.awvalid),  // input wire s_axi_awvalid
  .s_axi_awready(m_axi_input.awready),  // output wire s_axi_awready
  .s_axi_wdata  (m_axi_input.wdata),      // input wire [31 : 0] s_axi_wdata
  .s_axi_wstrb  (m_axi_input.wstrb),      // input wire [3 : 0] s_axi_wstrb
  .s_axi_wlast  (m_axi_input.wlast),      // input wire s_axi_wlast
  .s_axi_wvalid (m_axi_input.wvalid),    // input wire s_axi_wvalid
  .s_axi_wready (m_axi_input.wready),    // output wire s_axi_wready
  .s_axi_bid    (m_axi_input.bid),          // output wire [3 : 0] s_axi_bid
  .s_axi_bresp  (m_axi_input.bresp),      // output wire [1 : 0] s_axi_bresp
  .s_axi_bvalid (m_axi_input.bvalid),    // output wire s_axi_bvalid
  .s_axi_bready (m_axi_input.bready),    // input wire s_axi_bready
  .s_axi_arid   (m_axi_input.arid),        // input wire [3 : 0] s_axi_arid
  .s_axi_araddr (m_axi_input.araddr),    // input wire [31 : 0] s_axi_araddr
  .s_axi_arlen  (m_axi_input.arlen),      // input wire [7 : 0] s_axi_arlen
  .s_axi_arsize (m_axi_input.arsize),    // input wire [2 : 0] s_axi_arsize
  .s_axi_arburst(m_axi_input.arburst),  // input wire [1 : 0] s_axi_arburst
  .s_axi_arvalid(m_axi_input.arvalid),  // input wire s_axi_arvalid
  .s_axi_arready(m_axi_input.arready),  // output wire s_axi_arready
  .s_axi_rid    (m_axi_input.rid),          // output wire [3 : 0] s_axi_rid
  .s_axi_rdata  (m_axi_input.rdata),      // output wire [31 : 0] s_axi_rdata
  .s_axi_rresp  (m_axi_input.rresp),      // output wire [1 : 0] s_axi_rresp
  .s_axi_rlast  (m_axi_input.rlast),      // output wire s_axi_rlast
  .s_axi_rvalid (m_axi_input.rvalid),    // output wire s_axi_rvalid
  .s_axi_rready (m_axi_input.rready)    // input wire s_axi_rready
);

 mobilenet_ip mobilenet_ip (
    .s_axi_AXILiteS_AWADDR(0),      // input wire [6 : 0] s_axi_AXILiteS_AWADDR
    .s_axi_AXILiteS_AWVALID(0),    // input wire s_axi_AXILiteS_AWVALID
    .s_axi_AXILiteS_AWREADY(),    // output wire s_axi_AXILiteS_AWREADY
    .s_axi_AXILiteS_WDATA(0),        // input wire [31 : 0] s_axi_AXILiteS_WDATA
    .s_axi_AXILiteS_WSTRB(0),        // input wire [3 : 0] s_axi_AXILiteS_WSTRB
    .s_axi_AXILiteS_WVALID(0),      // input wire s_axi_AXILiteS_WVALID
    .s_axi_AXILiteS_WREADY(),      // output wire s_axi_AXILiteS_WREADY
    .s_axi_AXILiteS_BRESP(),        // output wire [1 : 0] s_axi_AXILiteS_BRESP
    .s_axi_AXILiteS_BVALID(),      // output wire s_axi_AXILiteS_BVALID
    .s_axi_AXILiteS_BREADY(0),      // input wire s_axi_AXILiteS_BREADY
    .s_axi_AXILiteS_ARADDR(0),      // input wire [6 : 0] s_axi_AXILiteS_ARADDR
    .s_axi_AXILiteS_ARVALID(0),    // input wire s_axi_AXILiteS_ARVALID
    .s_axi_AXILiteS_ARREADY(),    // output wire s_axi_AXILiteS_ARREADY
    .s_axi_AXILiteS_RDATA(),        // output wire [31 : 0] s_axi_AXILiteS_RDATA
    .s_axi_AXILiteS_RRESP(),        // output wire [1 : 0] s_axi_AXILiteS_RRESP
    .s_axi_AXILiteS_RVALID(),      // output wire s_axi_AXILiteS_RVALID
    .s_axi_AXILiteS_RREADY(0),      // input wire s_axi_AXILiteS_RREADY
    .ap_clk(clk),                                    // input wire ap_clk
    .ap_rst_n(rstn),                                // input wire ap_rst_n
    .ap_start(wr_data_done),                                // input wire ap_start
    .ap_done(),                                  // output wire ap_done
    .ap_idle(),                                  // output wire ap_idle
    .ap_ready(),                                // output wire ap_ready
    .m_axi_IMG_AWADDR       (),                // output wire [31 : 0] m_axi_IMG_AWADDR
    .m_axi_IMG_AWLEN        (),                  // output wire [7 : 0] m_axi_IMG_AWLEN
    .m_axi_IMG_AWSIZE       (),                // output wire [2 : 0] m_axi_IMG_AWSIZE
    .m_axi_IMG_AWBURST      (),              // output wire [1 : 0] m_axi_IMG_AWBURST
    .m_axi_IMG_AWLOCK       (),                // output wire [1 : 0] m_axi_IMG_AWLOCK
    .m_axi_IMG_AWREGION     (),            // output wire [3 : 0] m_axi_IMG_AWREGION
    .m_axi_IMG_AWCACHE      (),              // output wire [3 : 0] m_axi_IMG_AWCACHE
    .m_axi_IMG_AWPROT       (),                // output wire [2 : 0] m_axi_IMG_AWPROT
    .m_axi_IMG_AWQOS        (),                  // output wire [3 : 0] m_axi_IMG_AWQOS
    .m_axi_IMG_AWVALID      (),              // output wire m_axi_IMG_AWVALID
    .m_axi_IMG_AWREADY      (1),              // input wire m_axi_IMG_AWREADY
    .m_axi_IMG_WDATA        (),                  // output wire [31 : 0] m_axi_IMG_WDATA
    .m_axi_IMG_WSTRB        (),                  // output wire [3 : 0] m_axi_IMG_WSTRB
    .m_axi_IMG_WLAST        (),                  // output wire m_axi_IMG_WLAST
    .m_axi_IMG_WVALID       (),                // output wire m_axi_IMG_WVALID
    .m_axi_IMG_WREADY       (1),                // input wire m_axi_IMG_WREADY
    .m_axi_IMG_BRESP        (3),                  // input wire [1 : 0] m_axi_IMG_BRESP
    .m_axi_IMG_BVALID       (1),                // input wire m_axi_IMG_BVALID
    .m_axi_IMG_BREADY       (),                // output wire m_axi_IMG_BREADY
    .m_axi_IMG_ARADDR       (),                // output wire [31 : 0] m_axi_IMG_ARADDR
    .m_axi_IMG_ARLEN        (m_axi_img.arlen),                  // output wire [7 : 0] m_axi_IMG_ARLEN
    .m_axi_IMG_ARSIZE       (m_axi_img.arsize),                // output wire [2 : 0] m_axi_IMG_ARSIZE
    .m_axi_IMG_ARBURST      (m_axi_img.arburst),              // output wire [1 : 0] m_axi_IMG_ARBURST
    .m_axi_IMG_ARLOCK       (m_axi_img.arlock),                // output wire [1 : 0] m_axi_IMG_ARLOCK
    .m_axi_IMG_ARREGION     (m_axi_img.arregion),            // output wire [3 : 0] m_axi_IMG_ARREGION
    .m_axi_IMG_ARCACHE      (m_axi_img.arcache),              // output wire [3 : 0] m_axi_IMG_ARCACHE
    .m_axi_IMG_ARPROT       (m_axi_img.arprot),                // output wire [2 : 0] m_axi_IMG_ARPROT
    .m_axi_IMG_ARQOS        (m_axi_img.arqos),                  // output wire [3 : 0] m_axi_IMG_ARQOS
    .m_axi_IMG_ARVALID      (m_axi_img.arvalid),              // output wire m_axi_IMG_ARVALID
    .m_axi_IMG_ARREADY      (m_axi_img.arready),              // input wire m_axi_IMG_ARREADY
    .m_axi_IMG_RDATA        (m_axi_img.rdata),                  // input wire [31 : 0] m_axi_IMG_RDATA
    .m_axi_IMG_RRESP        (m_axi_img.rresp),                  // input wire [1 : 0] m_axi_IMG_RRESP
    .m_axi_IMG_RLAST        (m_axi_img.rlast),                  // input wire m_axi_IMG_RLAST
    .m_axi_IMG_RVALID       (m_axi_img.rvalid),                // input wire m_axi_IMG_RVALID
    .m_axi_IMG_RREADY       (m_axi_img.rready),                // output wire m_axi_IMG_RREADY
    .m_axi_INPUT_r_AWADDR   (m_axi_input.awaddr),        // output wire [31 : 0] m_axi_INPUT_r_AWADDR
    .m_axi_INPUT_r_AWLEN    (m_axi_input.awlen),          // output wire [7 : 0] m_axi_INPUT_r_AWLEN
    .m_axi_INPUT_r_AWSIZE   (m_axi_input.awsize),        // output wire [2 : 0] m_axi_INPUT_r_AWSIZE
    .m_axi_INPUT_r_AWBURST  (m_axi_input.awburst),      // output wire [1 : 0] m_axi_INPUT_r_AWBURST
    .m_axi_INPUT_r_AWLOCK   (m_axi_input.awlock),        // output wire [1 : 0] m_axi_INPUT_r_AWLOCK
    .m_axi_INPUT_r_AWREGION (m_axi_input.awregion),    // output wire [3 : 0] m_axi_INPUT_r_AWREGION
    .m_axi_INPUT_r_AWCACHE  (m_axi_input.awcache),      // output wire [3 : 0] m_axi_INPUT_r_AWCACHE
    .m_axi_INPUT_r_AWPROT   (m_axi_input.awprot),        // output wire [2 : 0] m_axi_INPUT_r_AWPROT
    .m_axi_INPUT_r_AWQOS    (m_axi_input.awqos),          // output wire [3 : 0] m_axi_INPUT_r_AWQOS
    .m_axi_INPUT_r_AWVALID  (m_axi_input.awvalid),      // output wire m_axi_INPUT_r_AWVALID
    .m_axi_INPUT_r_AWREADY  (m_axi_input.awready),      // input wire m_axi_INPUT_r_AWREADY
    .m_axi_INPUT_r_WDATA    (m_axi_input.wdata),          // output wire [31 : 0] m_axi_INPUT_r_WDATA
    .m_axi_INPUT_r_WSTRB    (m_axi_input.wstrb),          // output wire [3 : 0] m_axi_INPUT_r_WSTRB
    .m_axi_INPUT_r_WLAST    (m_axi_input.wlast),          // output wire m_axi_INPUT_r_WLAST
    .m_axi_INPUT_r_WVALID   (m_axi_input.wvalid),        // output wire m_axi_INPUT_r_WVALID
    .m_axi_INPUT_r_WREADY   (m_axi_input.wready),        // input wire m_axi_INPUT_r_WREADY
    .m_axi_INPUT_r_BRESP    (m_axi_input.bresp),          // input wire [1 : 0] m_axi_INPUT_r_BRESP
    .m_axi_INPUT_r_BVALID   (m_axi_input.bvalid),        // input wire m_axi_INPUT_r_BVALID
    .m_axi_INPUT_r_BREADY   (m_axi_input.bready),        // output wire m_axi_INPUT_r_BREADY
    .m_axi_INPUT_r_ARADDR   (m_axi_input.araddr),        // output wire [31 : 0] m_axi_INPUT_r_ARADDR
    .m_axi_INPUT_r_ARLEN    (m_axi_input.arlen),          // output wire [7 : 0] m_axi_INPUT_r_ARLEN
    .m_axi_INPUT_r_ARSIZE   (m_axi_input.arsize),        // output wire [2 : 0] m_axi_INPUT_r_ARSIZE
    .m_axi_INPUT_r_ARBURST  (m_axi_input.arburst),      // output wire [1 : 0] m_axi_INPUT_r_ARBURST
    .m_axi_INPUT_r_ARLOCK   (m_axi_input.arlock),        // output wire [1 : 0] m_axi_INPUT_r_ARLOCK
    .m_axi_INPUT_r_ARREGION (m_axi_input.arregion),    // output wire [3 : 0] m_axi_INPUT_r_ARREGION
    .m_axi_INPUT_r_ARCACHE  (m_axi_input.arcache),      // output wire [3 : 0] m_axi_INPUT_r_ARCACHE
    .m_axi_INPUT_r_ARPROT   (m_axi_input.arprot),        // output wire [2 : 0] m_axi_INPUT_r_ARPROT
    .m_axi_INPUT_r_ARQOS    (m_axi_input.arqos),          // output wire [3 : 0] m_axi_INPUT_r_ARQOS
    .m_axi_INPUT_r_ARVALID  (m_axi_input.arvalid),      // output wire m_axi_INPUT_r_ARVALID
    .m_axi_INPUT_r_ARREADY  (m_axi_input.arready),      // input wire m_axi_INPUT_r_ARREADY
    .m_axi_INPUT_r_RDATA    (m_axi_input.rdata),          // input wire [31 : 0] m_axi_INPUT_r_RDATA
    .m_axi_INPUT_r_RRESP    (m_axi_input.rresp),          // input wire [1 : 0] m_axi_INPUT_r_RRESP
    .m_axi_INPUT_r_RLAST    (m_axi_input.rlast),          // input wire m_axi_INPUT_r_RLAST
    .m_axi_INPUT_r_RVALID   (m_axi_input.rvalid),        // input wire m_axi_INPUT_r_RVALID
    .m_axi_INPUT_r_RREADY   (m_axi_input.rready),        // output wire m_axi_INPUT_r_RREADY
    .m_axi_OUTPUT_r_AWADDR  (),      // output wire [31 : 0] m_axi_OUTPUT_r_AWADDR
    .m_axi_OUTPUT_r_AWLEN   (),        // output wire [7 : 0] m_axi_OUTPUT_r_AWLEN
    .m_axi_OUTPUT_r_AWSIZE  (),      // output wire [2 : 0] m_axi_OUTPUT_r_AWSIZE
    .m_axi_OUTPUT_r_AWBURST (),    // output wire [1 : 0] m_axi_OUTPUT_r_AWBURST
    .m_axi_OUTPUT_r_AWLOCK  (),      // output wire [1 : 0] m_axi_OUTPUT_r_AWLOCK
    .m_axi_OUTPUT_r_AWREGION(),  // output wire [3 : 0] m_axi_OUTPUT_r_AWREGION
    .m_axi_OUTPUT_r_AWCACHE (),    // output wire [3 : 0] m_axi_OUTPUT_r_AWCACHE
    .m_axi_OUTPUT_r_AWPROT  (),      // output wire [2 : 0] m_axi_OUTPUT_r_AWPROT
    .m_axi_OUTPUT_r_AWQOS   (),        // output wire [3 : 0] m_axi_OUTPUT_r_AWQOS
    .m_axi_OUTPUT_r_AWVALID (output_awvalid),    // output wire m_axi_OUTPUT_r_AWVALID
    .m_axi_OUTPUT_r_AWREADY (output_awready),    // input wire m_axi_OUTPUT_r_AWREADY
    .m_axi_OUTPUT_r_WDATA   (output_wdata),        // output wire [31 : 0] m_axi_OUTPUT_r_WDATA
    .m_axi_OUTPUT_r_WSTRB   (),        // output wire [3 : 0] m_axi_OUTPUT_r_WSTRB
    .m_axi_OUTPUT_r_WLAST   (output_wlast),        // output wire m_axi_OUTPUT_r_WLAST
    .m_axi_OUTPUT_r_WVALID  (output_wvalid),      // output wire m_axi_OUTPUT_r_WVALID
    .m_axi_OUTPUT_r_WREADY  (output_wready),      // input wire m_axi_OUTPUT_r_WREADY
    .m_axi_OUTPUT_r_BRESP   (2'b11),        // input wire [1 : 0] m_axi_OUTPUT_r_BRESP
    .m_axi_OUTPUT_r_BVALID  (1),      // input wire m_axi_OUTPUT_r_BVALID
    .m_axi_OUTPUT_r_BREADY  (),      // output wire m_axi_OUTPUT_r_BREADY
    .m_axi_OUTPUT_r_ARADDR  (),      // output wire [31 : 0] m_axi_OUTPUT_r_ARADDR
    .m_axi_OUTPUT_r_ARLEN   (),        // output wire [7 : 0] m_axi_OUTPUT_r_ARLEN
    .m_axi_OUTPUT_r_ARSIZE  (),      // output wire [2 : 0] m_axi_OUTPUT_r_ARSIZE
    .m_axi_OUTPUT_r_ARBURST (),    // output wire [1 : 0] m_axi_OUTPUT_r_ARBURST
    .m_axi_OUTPUT_r_ARLOCK  (),      // output wire [1 : 0] m_axi_OUTPUT_r_ARLOCK
    .m_axi_OUTPUT_r_ARREGION(),  // output wire [3 : 0] m_axi_OUTPUT_r_ARREGION
    .m_axi_OUTPUT_r_ARCACHE (),    // output wire [3 : 0] m_axi_OUTPUT_r_ARCACHE
    .m_axi_OUTPUT_r_ARPROT  (),      // output wire [2 : 0] m_axi_OUTPUT_r_ARPROT
    .m_axi_OUTPUT_r_ARQOS   (),        // output wire [3 : 0] m_axi_OUTPUT_r_ARQOS
    .m_axi_OUTPUT_r_ARVALID (),    // output wire m_axi_OUTPUT_r_ARVALID
    .m_axi_OUTPUT_r_ARREADY (0),    // input wire m_axi_OUTPUT_r_ARREADY
    .m_axi_OUTPUT_r_RDATA   (0),        // input wire [31 : 0] m_axi_OUTPUT_r_RDATA
    .m_axi_OUTPUT_r_RRESP   (0),        // input wire [1 : 0] m_axi_OUTPUT_r_RRESP
    .m_axi_OUTPUT_r_RLAST   (0),        // input wire m_axi_OUTPUT_r_RLAST
    .m_axi_OUTPUT_r_RVALID  (0),      // input wire m_axi_OUTPUT_r_RVALID
    .m_axi_OUTPUT_r_RREADY  ()      // output wire m_axi_OUTPUT_r_RREADY
  );


	wire 					output_awvalid;
	wire 					output_awready;
	wrie 					output_wvalid;
	wire 					output_wready;
	wire 					output_wlast;	
	wire[31:0]				output_wdata;

	reg						tx_metadata_valid;
	reg 					tx_data_valid;
	reg[511:0]				tx_data_data;	
	reg[7:0]				i;			


	assign output_awready = 1;
	assign output_wready = 1;

	always@(posedge clk)begin
		if(~rstn)begin
			tx_metadata_valid				<= 1'b0;
		end
		else if(output_awvalid & output_awready)begin
			tx_metadata_valid				<= 1'b1;
		end
		else if(m_axis_tx_metadata.valid & m_axis_tx_metadata.ready)begin
			tx_metadata_valid				<= 1'b0;
		end
		else begin
			tx_metadata_valid				<= tx_metadata_valid;
		end
	end

	assign m_axis_tx_metadata.valid = tx_metadata_valid;
	assign m_axis_tx_metadata.data = {32'h40,session_id};

	always@(posedge clk)begin
		if(~rstn)begin
			i								<= 0;
		end
		else if(output_wlast & output_wready & output_wvalid)begin
			i  								<= 0;
		end
		else if(output_wready & output_wvalid)begin
			i 								<= i + 1;
		end
		else begin
			i 								<= i;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			tx_data_data					<= 0;
		end
		else if(output_wready & output_wvalid)begin
			if(i==0)
				tx_data_data[31:0]			<= output_wdata;
			else if(i==1)
				tx_data_data[63:32]			<= output_wdata;
			else if(i==2)
				tx_data_data[95:64]			<= output_wdata;
			else if(i==3)
				tx_data_data[127:96]			<= output_wdata;
			else if(i==4)
				tx_data_data[159:128]			<= output_wdata;
			else
				tx_data_data[191:160]			<= output_wdata;												
		end
		else begin
			tx_data_data					<= tx_data_data;
		end
	end

	always@(posedge clk)begin
		if(~rstn)begin
			tx_data_valid				<= 1'b0;
		end
		else if(output_wlast & output_wready & output_wvalid)begin
			tx_data_valid				<= 1'b1;
		end
		else if(m_axis_tx_data.valid & m_axis_tx_data.ready)begin
			tx_data_valid				<= 1'b0;
		end
		else begin
			tx_data_valid				<= tx_data_valid;
		end
	end



	assign m_axis_tx_data.valid = tx_data_valid;
	assign m_axis_tx_data.data = tx_data_data;
	assign m_axis_tx_data.keep = 64'hffff_ffff_ffff_ffff;
	assign m_axis_tx_data.last = 1;











  ////////////////////////////mpi time

  reg[31:0]                           time_counter;
  reg                                 time_en;
  reg[31:0]                            max_img,max_input;

  always@(posedge clk)begin
      if(~rstn)begin
          time_en                     <= 1'b0;
      end
      else if(wr_data_done)begin
          time_en                     <= 1'b1;
      end
      else if(tx_data_valid)begin
          time_en                     <= 1'b0;
      end        
      else begin
          time_en                     <= time_en;
      end
  end     

  always@(posedge clk)begin
      if(~rstn)begin
          time_counter                <= 1'b0;
      end
      else if(time_en)begin
          time_counter                <= time_counter + 1'b1;
      end
      else begin
          time_counter                <= time_counter;
      end
  end    





  assign status_reg = time_counter;





  
  ila_mobilenet ila_mobilenet_inst (
	.clk(clk), // input wire clk


	.probe0(m_axi_img.arready), // input wire [0:0]  probe0  
	.probe1(m_axi_img.arvalid), // input wire [0:0]  probe1 
	.probe2(m_axi_img.araddr), // input wire [31:0]  probe2 
	.probe3(m_axi_img.rready), // input wire [0:0]  probe3 
	.probe4(m_axi_img.rvalid), // input wire [0:0]  probe4 
	.probe5(m_axi_img.arlen), // input wire [7:0]  probe5 
	.probe6(max_img), // input wire [31:0]  probe6 
	.probe7(m_axi_input.arready), // input wire [0:0]  probe7 
	.probe8(m_axi_input.arvalid), // input wire [0:0]  probe8 
	.probe9(m_axi_input.araddr), // input wire [31:0]  probe9 
	.probe10(m_axi_input.rready), // input wire [0:0]  probe10 
	.probe11(m_axi_input.rvalid), // input wire [0:0]  probe11 
	.probe12(m_axi_input.arlen), // input wire [7:0]  probe12 
	.probe13(time_counter), // inpwut wire [31:0]  probe13 
	.probe14(m_axi_output.awready), // input wire [0:0]  probe14 
	.probe15(m_axi_output.awvalid), // input wire [0:0]  probe15 
	.probe16(m_axi_output.awaddr), // input wire [31:0]  probe16 
	.probe17(m_axi_output.wready), // input wire [0:0]  probe17 
	.probe18(m_axi_output.wvalid), // input wire [0:0]  probe18 
	.probe19(m_axi_output.awlen), // input wire [7:0]  probe19 
	.probe20(max_input), // input wire [31:0]  probe20 
	.probe21(wr_data_done) // input wire [0:0]  probe21
);





 endmodule
 `default_nettype wire
 