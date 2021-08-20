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
 
 module send_inference (
    input wire         clk,
    input wire         rstn,

    //DMA Commands
    axis_mem_cmd.master         axis_dma_read_cmd,
    //DMA Data streams      
	axi_stream.slave            axis_dma_read_data,    
    
     
	//tcp send
    axis_meta.master     		m_axis_tx_metadata,
    axi_stream.master    		m_axis_tx_data,

	//tcp recv    
    axis_meta.slave    			s_axis_rx_metadata,
    axi_stream.slave   			s_axis_rx_data,

    // control reg
     input wire[14:0][31:0]      control_reg,
     output wire[7:0][31:0]      status_reg     
 
 );



    reg                             dma_read_valid;
    reg [63:0]                      dma_base_addr;
    reg [31:0]                      length;
    reg [15:0]                      session_id;
    reg                             start,start_r;

//////////////////////////////////////////////////dma cmd    
	always@(posedge clk)begin
		if(~rstn)begin
			dma_read_valid				<= 1'b0;
		end
		else if(start & ~start_r)begin
			dma_read_valid				<= 1'b1;
		end
		else if(axis_dma_read_cmd.valid & axis_dma_read_cmd.ready)begin
			dma_read_valid				<= 1'b0;
		end
		else begin
			dma_read_valid				<= dma_read_valid;
		end
	end

    assign axis_dma_read_cmd.valid = dma_read_valid;
    assign axis_dma_read_cmd.address = dma_base_addr;
    assign axis_dma_read_cmd.length = length;

/////////////////////////////////////////////////////////////////////////////tx meta
    reg                             tx_metadata_valid;

	always@(posedge clk)begin
		if(~rstn)begin
			tx_metadata_valid				<= 1'b0;
		end
		else if(start & ~start_r)begin
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
	assign m_axis_tx_metadata.data = {length,session_id};
    
///////////////////////////////////////////////////////////////////dma data
    reg[31:0]                           data_cnt;
    reg[31:0]                           data_cnt_minus;
    wire                                dma_read_data_last;

	always @(posedge clk)begin
		if(~rstn)begin
			data_cnt 						<= 1'b0;
		end
		else if(dma_read_data_last)begin
			data_cnt						<= 1'b0;
		end
		else if (axis_dma_read_data.ready & axis_dma_read_data.valid)begin
			data_cnt						<= data_cnt + 1'b1;
		end
		else begin
			data_cnt						<= data_cnt;
		end		
	end

    assign dma_read_data_last               = (data_cnt == data_cnt_minus) && axis_dma_read_data.ready && axis_dma_read_data.valid;

	axis_data_fifo_512_d4096 read_data_slice_fifo (
		.s_axis_aresetn(rstn),          // input wire s_axis_aresetn
		.s_axis_aclk(clk),                // input wire s_axis_aclk
		.s_axis_tvalid(axis_dma_read_data.valid),            // input wire s_axis_tvalid
		.s_axis_tready(axis_dma_read_data.ready),            // output wire s_axis_tready
		.s_axis_tdata(axis_dma_read_data.data),              // input wire [511 : 0] s_axis_tdata
		.s_axis_tkeep(axis_dma_read_data.keep),              // input wire [63 : 0] s_axis_tkeep
		.s_axis_tlast(dma_read_data_last),              // input wire s_axis_tlast
		.m_axis_tvalid(m_axis_tx_data.valid),            // output wire m_axis_tvalid
		.m_axis_tready(m_axis_tx_data.ready),            // input wire m_axis_tready
		.m_axis_tdata(m_axis_tx_data.data),              // output wire [511 : 0] m_axis_tdata
		.m_axis_tkeep(m_axis_tx_data.keep),              // output wire [63 : 0] m_axis_tkeep
		.m_axis_tlast(m_axis_tx_data.last),              // output wire m_axis_tlast
		.axis_wr_data_count(),  // output wire [31 : 0] axis_wr_data_count
		.axis_rd_data_count()  // output wire [31 : 0] axis_rd_data_count
	  );



always@(posedge clk)begin
    session_id                      <= control_reg[0];
    dma_base_addr                   <= {control_reg[2],control_reg[1]};
    start                           <= control_reg[3][0];
    start_r                         <= start;
    length							<= 32'h26400;
    data_cnt_minus					<= (length >>>6)-1;
end


//////////////////////////////////////////////////////////////RX

    assign s_axis_rx_metadata.ready     = 1;
    assign s_axis_rx_data.ready = 1;


  ////////////////////////////mpi time

  reg[31:0]                           time_counter;
  reg                                 time_en;
  reg[31:0]                            max_img,max_input;

  always@(posedge clk)begin
      if(~rstn)begin
          time_en                     <= 1'b0;
      end
      else if(s_axis_rx_data.ready & s_axis_rx_data.valid)begin
          time_en                     <= 1'b1;
      end
      else if(start & ~start_r)begin
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





  assign status_reg[0] = time_counter;










 endmodule
 `default_nettype wire
 