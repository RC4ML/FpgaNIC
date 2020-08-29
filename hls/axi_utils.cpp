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
#include "axi_utils.hpp"

bool checkIfResponse(ibOpCode code)
{
	return (code == RC_RDMA_READ_RESP_FIRST || code == RC_RDMA_READ_RESP_MIDDLE ||
			code == RC_RDMA_READ_RESP_LAST || code == RC_RDMA_READ_RESP_ONLY ||
			code == RC_ACK);
}

bool checkIfWriteOrPartReq(ibOpCode code)
{
	return (code == RC_RDMA_WRITE_FIRST || code == RC_RDMA_WRITE_MIDDLE ||
			code == RC_RDMA_WRITE_LAST || code == RC_RDMA_WRITE_ONLY ||
			code == RC_RDMA_PART_FIRST || code == RC_RDMA_PART_MIDDLE ||
			code == RC_RDMA_PART_LAST || code == RC_RDMA_PART_ONLY);
}

bool checkIfAethHeader(ibOpCode code)
{
	return (code == RC_RDMA_READ_RESP_ONLY || code == RC_RDMA_READ_RESP_FIRST ||
			code == RC_RDMA_READ_RESP_LAST || code == RC_ACK);
}

bool checkIfRethHeader(ibOpCode code)
{
	return (code == RC_RDMA_WRITE_ONLY || code == RC_RDMA_WRITE_FIRST ||
			code == RC_RDMA_PART_ONLY || code == RC_RDMA_PART_FIRST ||
			code == RC_RDMA_READ_REQUEST);
}

template <>
void assignDest<routedAxiWord>(routedAxiWord& d, routedAxiWord& s)
{
	d.dest = s.dest;
}

ap_uint<32> lenToKeep(ap_uint<6> length)
{
	switch (length)
	{
	case 1:
	    return 0x01;
	  case 2:
	    return 0x03;
	  case 3:
	    return 0x07;
	  case 4:
	    return 0x0F;
	  case 5:
	    return 0x1F;
	  case 6:
	    return 0x3F;
	  case 7:
	    return 0x7F;
	  case 8:
	    return 0xFF;
	  case 9:
		return 0x01FF;
	  case 10:
		return 0x03FF;
	  case 11:
		return 0x07FF;
	  case 12:
		return 0x0FFF;
	  case 13:
		return 0x1FFF;
	  case 14:
		return 0x3FFF;
	  case 15:
		return 0x7FFF;
	  case 16:
		return 0xFFFF;
	  case 17:
		return 0x01FFFF;
	  case 18:
		return 0x03FFFF;
	  case 19:
		return 0x07FFFF;
	  case 20:
		return 0x0FFFFF;
	  case 21:
		return 0x1FFFFF;
	  case 22:
		return 0x3FFFFF;
	  case 23:
		return 0x7FFFFF;
	  case 24:
		return 0xFFFFFF;
	  case 25:
		return 0x01FFFFFF;
	  case 26:
		return 0x03FFFFFF;
	  case 27:
		return 0x07FFFFFF;
	  case 28:
		return 0x0FFFFFFF;
	  case 29:
		return 0x1FFFFFFF;
	  case 30:
		return 0x3FFFFFFF;
	  case 31:
		return 0x7FFFFFFF;
	  case 32:
		return 0xFFFFFFFF;
	}
}

ap_uint<6> keepToLen(ap_uint<32> keepValue)
{
	switch (keepValue)
	{
	case 0x01:
		return 0x1;
	case 0x3:
		return 0x2;
	case 0x07:
		return 0x3;
	case 0x0F:
		return 0x4;
	case 0x1F:
		return 0x5;
	case 0x3F:
		return 0x6;
	case 0x7F:
		return 0x7;
	case 0xFF:
		return 0x8;
	}
}
