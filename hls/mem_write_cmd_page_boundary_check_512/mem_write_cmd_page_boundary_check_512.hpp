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
#ifndef MEM_WRITE_CMD_BOUNDARY_512
#define MEM_WRITE_CMD_BOUNDARY_512

#include "../axi_utils.hpp"
#include "../mem_utils.hpp"

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

const unsigned PAGE_SIZE 	= 65536;   //2MB 2097152    64KB 65536

const unsigned PAGE_BIT_WIDTH 	= ConstLog2(PAGE_SIZE);
const unsigned PAGE_OFFSET 		= ConstOffset(PAGE_BIT_WIDTH);


struct internalCmd
{
	ap_uint<48> addr;
	ap_uint<32> len;
	ap_uint<24> page_offset;
	ap_uint<24>	new_length;
	internalCmd() {}
	internalCmd(ap_uint<64> addr, ap_uint<32> len, ap_uint<24> po, ap_uint<24> nl)
		:addr(addr), len(len), page_offset(po), new_length(nl) {}
};

struct boundCheckMeta
{
	ap_uint<32> length;
	bool		isSplit;
	boundCheckMeta() {}
	boundCheckMeta(ap_uint<32> len, bool split)
		:length(len), isSplit(split) {}
};

void mem_write_cmd_page_boundary_check_512(	hls::stream<memCmd>&		cmdIn,
											hls::stream<memCmd>&		cmdOut,
											ap_uint<48>			regBaseVaddr);

#endif
