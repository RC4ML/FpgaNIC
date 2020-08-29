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
#include "mem_write_cmd_page_boundary_check_512.hpp"
#include <iostream>


int main()
{

	hls::stream<memCmd>		cmdIn("cmdIn");
	hls::stream<memCmd>		cmdOut("cmdOut");
	ap_uint<48>			regBaseVaddr;


	regBaseVaddr = 0x7f59dd400000;

	int numTuples = 1024*1024;
	cmdIn.write(memCmd(0x7f59de5ffaa8, numTuples*8));

	int count = 0;
	while (count < 1000)
	{
		mem_write_cmd_page_boundary_check_512(	cmdIn,
												cmdOut,
												regBaseVaddr);
		count++;
	}

	memCmd cmd;
	std::cout << "CMD" << std::endl;
	while(!cmdOut.empty())
	{
		cmdOut.read(cmd);
		std::cout << std::hex << "addr: " << cmd.addr << ", length: " << std::dec << cmd.len << std::endl;
	}


	return 0;
}
