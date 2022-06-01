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
#ifndef TLB_HPP
#define TLB_HPP

#include "../axi_utils.hpp"
#include "../mem_utils.hpp"

//#define USE_DDR


unsigned long ConstLog2(unsigned long val) {
  return val == 1 ? 0 : 1 + ConstLog2(val >> 1);
}

unsigned int ConstOffset(unsigned int val) {
	int res = 0;
	while(val--){
		res=res+(1<<val);
	}
  return res; 
}


const uint16_t TLB_ENTRIES 	= 512;
const unsigned PAGE_SIZE 	= 65536;   //2MB 2097152    64KB 65536

const unsigned PAGE_BIT_WIDTH 	= ConstLog2(PAGE_SIZE);
const unsigned PAGE_OFFSET 		= ConstOffset(PAGE_BIT_WIDTH);

struct dmaCmd
{
	ap_uint<64>  addr;
	ap_uint<28> len; //TODO increase length
	dmaCmd() {}
	dmaCmd(ap_int<64> addr, ap_uint<28> len)
		:addr(addr), len(len) {}
};

struct tlbMapping
{
	ap_uint<64> vaddr;
	ap_uint<64> paddr;
	bool		isBase;
	tlbMapping() {}
	tlbMapping(ap_uint<64> vaddr, ap_uint<64> paddr, bool isBase)
		:vaddr(vaddr), paddr(paddr), isBase(isBase) {}
};

struct tlbEntry
{
	ap_uint<48> paddr;
};


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
			ap_uint<32>&		regTlbMissCount,
			ap_uint<32>&		regPageCrossingCount);

#endif
