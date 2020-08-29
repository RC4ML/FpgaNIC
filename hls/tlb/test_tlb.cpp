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
#include <iostream>

int main()
{
	hls::stream<memCmd> 	s_axis_mem_write_cmd("s_axis_mem_write_cmd");
	hls::stream<memCmd> 	s_axis_mem_read_cmd("s_axis_mem_read_cmd");
	//FPGA on-board DDR
#ifdef USE_DDR
	hls::stream<mmCmd>		m_axis_ddr_write_cmd("m_axis_ddr_write_cmd");
	hls::stream<mmCmd>		m_axis_ddr_read_cmd("m_axis_ddr_read_cmd");
#endif
	//DMA
	hls::stream<dmaCmd>		m_axis_dma_write_cmd("m_axis_dma_write_cmd");
	hls::stream<dmaCmd>		m_axis_dma_read_cmd("m_axis_dma_read_cmd");
	//host interface
	hls::stream<tlbMapping> s_axis_tlb_interface("s_axis_tlb_interface");
	ap_uint<32>		regTlbMissCount;
	ap_uint<32>		regPageCrossingCount;


	s_axis_tlb_interface.write(tlbMapping(0xAABBCC00, 0x00C0000, true));
	s_axis_tlb_interface.write(tlbMapping(0xAADBCC00, 0x00D0000, false));

	int count = 0;
	while (count < 100)
	{
		if (count == 10)
		{
			s_axis_mem_write_cmd.write(memCmd(0x11, 0xAABBCC0F, 0x10));
		}
		if (count == 15)
		{
#ifdef USE_DDR
			s_axis_mem_read_cmd.write(memCmd(0x1, 0x8567, 0x40));
#endif
		}
		if (count == 20)
		{
			s_axis_mem_read_cmd.write(memCmd(0x11, 0xAADBCCDD, 0x20));
		}
		if (count == 25)
		{
#ifdef USE_DDR
			s_axis_mem_write_cmd.write(memCmd(0x1, 0xABCDEF, 0xA0));
#endif
		}

		tlb(	s_axis_mem_write_cmd,
				s_axis_mem_read_cmd,
#ifdef USE_DDR
				m_axis_ddr_write_cmd,
				m_axis_ddr_read_cmd,
#endif
				m_axis_dma_write_cmd,
				m_axis_dma_read_cmd,
				s_axis_tlb_interface,
				regTlbMissCount,
				regPageCrossingCount);
		count++;

	}
	int rc = 0;
	//DDR
#ifdef USE_DDR
	mmCmd cmd0;
	std::cout << "DDR WRITE" << std::endl;
	while (!m_axis_ddr_write_cmd.empty())
	{
		m_axis_ddr_write_cmd.read(cmd0);
		if (cmd0.saddr != 0xABCDEF || cmd0.bbt != 0xA0)
		{
			rc = -1;
			std::cerr << "[ERROR] ddr write" << std::endl;
		}
		std::cout << std::hex << "addr: " << cmd0.saddr << ", len: " << cmd0.bbt << std::endl;
	}
	std::cout << "DDR READ" << std::endl;
	while (!m_axis_ddr_read_cmd.empty())
	{
		m_axis_ddr_read_cmd.read(cmd0);
		if (cmd0.saddr != 0x8567 || cmd0.bbt != 0x40)
		{
			rc = -1;
			std::cerr << "[ERROR] ddr read" << std::endl;
		}
		std::cout << std::hex << "addr: " << cmd0.saddr << ", len: " << cmd0.bbt << std::endl;
	}
#endif
	//DMA
	dmaCmd cmd1;
	std::cout << "DMA WRITE" << std::endl;
	while (!m_axis_dma_write_cmd.empty())
	{
		m_axis_dma_write_cmd.read(cmd1);
		if (cmd1.addr != 0x00C000F || cmd1.len != 0x10)
		{
			rc = -1;
			std::cerr << "[ERROR] dma write" << std::endl;
		}
		std::cout << std::hex << "addr: " << cmd1.addr << ", len: " << cmd1.len << std::endl;
	}
	std::cout << "DMA READ" << std::endl;
	while (!m_axis_dma_read_cmd.empty())
	{
		m_axis_dma_read_cmd.read(cmd1);
		if (cmd1.addr != 0x00D00DD || cmd1.len != 0x20)
		{
			rc = -1;
			std::cerr << "[ERROR] dma read" << std::endl;
		}
		std::cout << std::hex << "addr: " << cmd1.addr << ", len: " << cmd1.len << std::endl;
	}

	std::cout << "Misses: " << regTlbMissCount << std::endl;
	std::cout << "Page crossings: " << regPageCrossingCount << std::endl;
	return rc;
}
