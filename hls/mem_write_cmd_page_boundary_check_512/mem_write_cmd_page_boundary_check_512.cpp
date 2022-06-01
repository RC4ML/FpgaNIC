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

void calculate_page_offset(	hls::stream<memCmd>&			cmdIn,
							hls::stream<internalCmd>&		cmdOut,
							ap_uint<48>						regBaseVaddr)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE=off

	if ((!cmdIn.empty()) & (!cmdOut.full()))
	{
		memCmd cmd = cmdIn.read();
		ap_uint<48> addr = cmd.addr - regBaseVaddr;
		ap_uint<24> page_offset = (addr & PAGE_OFFSET);
		ap_uint<24> newLength = PAGE_SIZE-page_offset;
		cmdOut.write(internalCmd(cmd.addr, cmd.len, page_offset, newLength));
	}
}

void boundary_check(hls::stream<internalCmd>&		cmdIn,
					hls::stream<memCmd>&			cmdOut)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE=off

	enum bcStateType{CMD, CMD_SPLIT};
	static bcStateType bc_state = CMD;
	static internalCmd cmd;


	static hls::stream<internalCmd> pageOffsetFifo("pageOffsetFifo");
	#pragma HLS stream depth=8 variable=pageOffsetFifo
	#pragma HLS DATA_PACK variable=pageOffsetFifo



	switch (bc_state)
	{
	case CMD:
		if (!cmdIn.empty())
		{
			cmdIn.read(cmd);
			if (cmd.page_offset + cmd.len > PAGE_SIZE)
			{
				cmdOut.write(memCmd(cmd.addr, cmd.new_length));
				cmd.addr += cmd.new_length;
				cmd.len -= cmd.new_length;
				bc_state = CMD_SPLIT;
			}
			else
			{
				cmdOut.write(memCmd(cmd.addr, cmd.len));
			}
		}
		break;
	case CMD_SPLIT:
		//TODO handle multiple splits
		if (cmd.len > PAGE_SIZE)
		{
			cmdOut.write(memCmd(cmd.addr, PAGE_SIZE));
			cmd.addr += PAGE_SIZE;
			cmd.len -= PAGE_SIZE;
			bc_state = CMD_SPLIT;
		}
		else
		{
			cmdOut.write(memCmd(cmd.addr, cmd.len));
			bc_state = CMD;
		}
		break;
	}
}




void mem_write_cmd_page_boundary_check_512(	hls::stream<memCmd>&			cmdIn,
											hls::stream<memCmd>&			cmdOut,
											ap_uint<48>						regBaseVaddr)
{
	#pragma HLS DATAFLOW disable_start_propagation
	#pragma HLS INTERFACE ap_ctrl_none port=return

	#pragma HLS INTERFACE axis register port=cmdIn name=s_axis_cmd
	#pragma HLS INTERFACE axis register port=cmdOut name=m_axis_cmd
	#pragma HLS DATA_PACK variable=cmdIn
	#pragma HLS DATA_PACK variable=cmdOut
	#pragma HLS INTERFACE ap_none port=regBaseVaddr

	static hls::stream<internalCmd> pageOffsetFifo("pageOffsetFifo");
	#pragma HLS stream depth=8 variable=pageOffsetFifo
	#pragma HLS DATA_PACK variable=pageOffsetFifo

	calculate_page_offset(cmdIn, pageOffsetFifo, regBaseVaddr);
	boundary_check(pageOffsetFifo, cmdOut);

}
