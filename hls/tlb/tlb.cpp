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
#include "tlb.hpp"




void tlb(	hls::stream<memCmd>& 	s_axis_mem_write_cmd,
			hls::stream<memCmd>& 	s_axis_mem_read_cmd,
			//FPGA on-board DDR
#ifdef USE_DDR
			hls::stream<mmCmd>&		m_axis_ddr_write_cmd,
			hls::stream<mmCmd>&		m_axis_ddr_read_cmd,
#endif
			//DMA
			hls::stream<dmaCmd>&		m_axis_dma_write_cmd,
			hls::stream<dmaCmd>&		m_axis_dma_read_cmd,
			//host interface
			hls::stream<tlbMapping>& s_axis_tlb_interface,
			//debug out
			ap_uint<32>&		regTlbMissCount,
			ap_uint<32>&		regPageCrossingCount)
{
#pragma HLS PIPELINE II=1
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE axis register port=s_axis_mem_write_cmd
#pragma HLS INTERFACE axis register port=s_axis_mem_read_cmd
#ifdef USE_DDR
#pragma HLS INTERFACE axis register port=m_axis_ddr_write_cmd
#pragma HLS INTERFACE axis register port=m_axis_ddr_read_cmd
#endif
#pragma HLS INTERFACE axis register port=m_axis_dma_write_cmd
#pragma HLS INTERFACE axis register port=m_axis_dma_read_cmd
#pragma HLS INTERFACE axis register port=s_axis_tlb_interface

#pragma HLS DATA_PACK variable=s_axis_mem_write_cmd
#pragma HLS DATA_PACK variable=s_axis_mem_read_cmd
#ifdef USE_DDR
#pragma HLS DATA_PACK variable=m_axis_ddr_write_cmd
#pragma HLS DATA_PACK variable=m_axis_ddr_read_cmd
#endif
#pragma HLS DATA_PACK variable=m_axis_dma_write_cmd
#pragma HLS DATA_PACK variable=m_axis_dma_read_cmd
#pragma HLS DATA_PACK variable=s_axis_tlb_interface

#pragma HLS INTERFACE ap_vld port=regTlbMissCount
#pragma HLS INTERFACE ap_vld port=regPageCrossingCount



	static tlbEntry tlb_table[TLB_ENTRIES];
	static ap_uint<64> base_vaddr = 0;
	static ap_uint<32> top_page_base = 0;
	static ap_uint<32> tlbMissCounter = 0;
	static ap_uint<32> tlbPageCrossingCounter = 0;

	memCmd cmd;
	tlbMapping newMapping;
	ap_uint<64> pbase;

	//TODO priority?
	if (!s_axis_mem_write_cmd.empty())
	{
		s_axis_mem_write_cmd.read(cmd);
		ap_uint<64> addr = cmd.addr - base_vaddr;
		ap_uint<64> page_base = (addr >> 21);
		ap_uint<64> page_offset = (addr & 0x1FFFFF);
		pbase = tlb_table[page_base].paddr;
		m_axis_dma_write_cmd.write(dmaCmd(pbase+page_offset, cmd.len));
		if (page_base > top_page_base)
		{
			tlbMissCounter++;
			regTlbMissCount = tlbMissCounter;
		}
		if (page_offset + cmd.len > (2*1024*1024))
		{
			tlbPageCrossingCounter++;
			regPageCrossingCount = tlbPageCrossingCounter;
		}

	}
	else if (!s_axis_mem_read_cmd.empty())
	{
		s_axis_mem_read_cmd.read(cmd);
		ap_uint<64> addr = cmd.addr - base_vaddr;
		ap_uint<64> page_base = (addr >> 21);
		ap_uint<64> page_offset = (addr & 0x1FFFFF);
		std::cout << "page_base "<< std::hex << page_base << std::endl;
		std::cout << "page_offset " << std::hex << page_offset << std::endl;
		pbase = tlb_table[page_base].paddr;
		m_axis_dma_read_cmd.write(dmaCmd(pbase+page_offset, cmd.len));
		if (page_base > top_page_base)
		{
			tlbMissCounter++;
			regTlbMissCount = tlbMissCounter;
		}
		if (page_offset + cmd.len > (2*1024*1024))
		{
			tlbPageCrossingCounter++;
			regPageCrossingCount = tlbPageCrossingCounter;
		}
	}
	else if (!s_axis_tlb_interface.empty())
	{
		s_axis_tlb_interface.read(newMapping);
		if (newMapping.isBase)
		{
			base_vaddr = newMapping.vaddr;
		}
		ap_uint<64> addr = newMapping.vaddr - base_vaddr;
		ap_uint<64> page_base = (addr >> 21);
		//ap_uint<64> page_offset = (addr & 0x1FFFFFF);
		tlb_table[page_base].paddr = newMapping.paddr;
		if (page_base > top_page_base)
		{
			top_page_base = page_base;
		}
	}
}
