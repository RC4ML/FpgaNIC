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
 
 module hyperloglog (
     input wire      clk,
     input wire      rstn,
    
     
     /* DMA INTERFACE */
    axis_mem_cmd.master     m_axis_dma_write_cmd,
 
    axi_stream.slave        s_axis_read_data,
    axi_stream.master       m_axis_dma_write_data,

    axis_meta.master        m_axis_tx_metadata,
    axi_stream.master       m_axis_tx_data,
    axis_meta.slave         s_axis_tx_status,    
 
    axis_meta.slave         s_axis_rx_metadata, 


    //control reg
     input wire[15:0][31:0]      control_reg,
     output wire[1:0][31:0]      status_reg     
 
 );

reg[63:0]                      dma_write_base_addr;
reg[31:0]                      length;
reg                            start,start_r,start_hyperloglog;
reg                            dma_read_cmd_valid;
reg                             tx_valid,tx_data_valid;                            
 
reg[31:0]                       dma_data_cnt,dma_data_cnt_minus;
wire                             dma_read_data_last;


reg[31:0]                       tx_data_cnt,tx_data_cnt_minus;
reg[15:0]                       session_id;

 always@(posedge clk)begin
    start                       <= control_reg[4][0];
    start_r                     <= start;
    start_hyperloglog           <= start & ~start_r; 
    dma_write_base_addr         <= {control_reg[1],control_reg[0]};
    length                      <= control_reg[2];
    session_id                  <= control_reg[3][15:0];
    dma_data_cnt_minus          <= (length >>>6) - 1;
    tx_data_cnt_minus           <= (length >>>2) - 16;
end




always@(posedge clk)begin
    if(~rstn)begin
        tx_valid            <= 0;
    end
    else if(start_hyperloglog)begin
        tx_valid            <= 1;
    end
    else if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
        tx_valid            <= 0;
    end
    else begin
        tx_valid            <= tx_valid;
    end
end 

assign s_axis_tx_status.ready = 1;
assign s_axis_rx_metadata.ready = 1;
assign m_axis_tx_metadata.valid = tx_valid;
assign m_axis_tx_metadata.data = {length,session_id};

assign m_axis_tx_data.valid = tx_data_valid;
assign m_axis_tx_data.data = {tx_data_cnt+15,tx_data_cnt+14,tx_data_cnt+13,tx_data_cnt+12,tx_data_cnt+11,tx_data_cnt+10,tx_data_cnt+9,tx_data_cnt+8,tx_data_cnt+7,tx_data_cnt+6,tx_data_cnt+5,tx_data_cnt+4,tx_data_cnt+3,tx_data_cnt+2,tx_data_cnt+1,tx_data_cnt};
assign m_axis_tx_data.keep = 64'hffff_ffff_ffff_ffff;
assign m_axis_tx_data.last = (tx_data_cnt == tx_data_cnt_minus) & m_axis_tx_data.valid & m_axis_tx_data.ready;


always@(posedge clk)begin
    if(~rstn)begin
        tx_data_valid            <= 0;
    end
    else if(m_axis_tx_data.last)begin
        tx_data_valid            <= 0;
    end
    else if(m_axis_tx_metadata.ready & m_axis_tx_metadata.valid)begin
        tx_data_valid            <= 1;
    end
    else begin
        tx_data_valid            <= tx_data_valid;
    end
end 


always@(posedge clk)begin
    if(~rstn)begin
        tx_data_cnt        <= 0;
    end
    else if(m_axis_tx_data.last)begin
        tx_data_cnt       <= 0;
    end
    else if(m_axis_tx_data.ready & m_axis_tx_data.valid)begin
        tx_data_cnt       <= tx_data_cnt +16;
    end
    else begin
        tx_data_cnt       <= tx_data_cnt;
    end
end 




always@(posedge clk)begin
    if(~rstn)begin
        dma_data_cnt        <= 0;
    end
    else if(dma_read_data_last)begin
        dma_data_cnt       <= 0;
    end
    else if(s_axis_read_data.ready & s_axis_read_data.valid)begin
        dma_data_cnt       <= dma_data_cnt +1;
    end
    else begin
        dma_data_cnt       <= dma_data_cnt;
    end
end 

assign dma_read_data_last = (dma_data_cnt == dma_data_cnt_minus) & s_axis_read_data.valid & s_axis_read_data.ready;

hyperloglog_ip hyperloglog_ip (
   .s_axis_input_tuple_TVALID      (s_axis_read_data.valid),  // input wire s_axis_input_tuple_TVALID
   .s_axis_input_tuple_TREADY      (s_axis_read_data.ready),  // output wire s_axis_input_tuple_TREADY
   .s_axis_input_tuple_TDATA       (s_axis_read_data.data),    // input wire [511 : 0] s_axis_input_tuple_TDATA
   .s_axis_input_tuple_TKEEP       (s_axis_read_data.keep),    // input wire [63 : 0] s_axis_input_tuple_TKEEP
   .s_axis_input_tuple_TLAST       (dma_read_data_last),    // input wire [0 : 0] s_axis_input_tuple_TLAST
   .m_axis_write_cmd_V_TVALID      (m_axis_dma_write_cmd.valid),  // output wire m_axis_write_cmd_V_TVALID
   .m_axis_write_cmd_V_TREADY      (m_axis_dma_write_cmd.ready),  // input wire m_axis_write_cmd_V_TREADY
   .m_axis_write_cmd_V_TDATA       ({m_axis_dma_write_cmd.length, m_axis_dma_write_cmd.address}),    // output wire [95 : 0] m_axis_write_cmd_V_TDATA
   .m_axis_write_data_TVALID       (m_axis_dma_write_data.valid),    // output wire m_axis_write_data_TVALID
   .m_axis_write_data_TREADY       (m_axis_dma_write_data.ready),    // input wire m_axis_write_data_TREADY
   .m_axis_write_data_TDATA        (m_axis_dma_write_data.data[31:0]),      // output wire [31 : 0] m_axis_write_data_TDATA
   .m_axis_write_data_TKEEP        (m_axis_dma_write_data.keep[3:0]),      // output wire [3 : 0] m_axis_write_data_TKEEP
   .m_axis_write_data_TLAST        (m_axis_dma_write_data.last),      // output wire [0 : 0] m_axis_write_data_TLAST
   .regBaseAddr_V                  (dma_write_base_addr),                          // input wire [63 : 0] regBaseAddr_V
   .ap_clk                         (clk),                                        // input wire ap_clk
   .ap_rst_n                       (rstn)                                    // input wire ap_rst_n
 );


assign m_axis_dma_write_data.data[511:32] = 0; 
assign m_axis_dma_write_data.keep[63:4] = 0;



///////////////////////////debug//////////////////////////////

reg[31:0]                               th_cnt;
reg 									th_en;

always @(posedge clk)begin
	if(~rstn)begin
		th_en 						    <= 1'b0;
    end
	else if(s_axis_read_data.ready & s_axis_read_data.valid)begin
		th_en						    <= 1'b1;
	end
	else if(m_axis_dma_write_cmd.ready & m_axis_dma_write_cmd.valid)begin
		th_en						    <= 1'b0;
    end    
	else begin
		th_en						    <= th_en;
	end		
end 

always @(posedge clk)begin
	if(~rstn)begin
		th_cnt 						    <= 1'b0;
    end
	else if(th_en)begin
		th_cnt						    <= th_cnt + 1;
	end   
	else begin
		th_cnt						    <= th_cnt;
	end		
end 


reg[31:0]                               net_cnt;
reg 									net_en;

always @(posedge clk)begin
	if(~rstn)begin
		net_en 						    <= 1'b0;
    end
	else if(dma_read_data_last)begin
		net_en						    <= 1'b0;
    end     
	else if(s_axis_read_data.ready & s_axis_read_data.valid)begin
		net_en						    <= 1'b1;
	end   
	else begin
		net_en						    <= net_en;
	end		
end 

always @(posedge clk)begin
	if(~rstn)begin
		net_cnt 						    <= 1'b0;
    end
	else if(net_en)begin
		net_cnt						    <= net_cnt + 1;
	end   
	else begin
		net_cnt						    <= net_cnt;
	end		
end 


assign status_reg[0] = th_cnt;
assign status_reg[1] = net_cnt;



//ila_hyperloglog inst_ila_hyperloglog (
//	.clk(clk), // input wire clk


//	.probe0(m_axis_tx_metadata.ready), // input wire [0:0]  probe0  
//	.probe1(m_axis_tx_metadata.valid), // input wire [0:0]  probe1 
//	.probe2(s_axis_read_data.data), // input wire [47:0]  probe2 
//	.probe3(m_axis_tx_data.ready), // input wire [0:0]  probe3 
//	.probe4(m_axis_tx_data.valid), // input wire [0:0]  probe4 
//	.probe5(m_axis_tx_data.last), // input wire [0:0]  probe5 
//	.probe6(s_axis_read_data.ready), // input wire [0:0]  probe6 
//	.probe7(s_axis_read_data.valid), // input wire [0:0]  probe7 
//	.probe8(dma_read_data_last), // input wire [0:0]  probe8 
//	.probe9(m_axis_dma_write_cmd.valid), // input wire [0:0]  probe9 
//	.probe10(m_axis_dma_write_cmd.ready), // input wire [0:0]  probe10 
//	.probe11(m_axis_dma_write_data.valid), // input wire [0:0]  probe11 
//	.probe12(m_axis_dma_write_data.ready), // input wire [0:0]  probe12 
//	.probe13(m_axis_dma_write_data.data[31:0]), // input wire [31:0]  probe13 
//	.probe14(m_axis_dma_write_data.last), // input wire [0:0]  probe14
//	.probe15(th_cnt), // input wire [31:0]  probe15 
//	.probe16(net_cnt) // input wire [0:0]  probe16    
//);
 
 endmodule
 `default_nettype wire
 