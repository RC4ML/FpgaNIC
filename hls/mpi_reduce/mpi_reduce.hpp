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
#ifndef MPI_REDUCE_HPP
#define MPI_REDUCE_HPP

#include "../axi_utils.hpp"
#include "../mem_utils.hpp"

//#define USE_DDR

struct mpiReduceInfo
{
	ap_uint<32>	ip_address;
	ap_uint<16>	sessionID;
};


struct mmCmd
{
	ap_uint<32>	length;
	ap_uint<64>	address;	

	mmCmd() {}
	mmCmd(ap_uint<64> addr, ap_uint<32> len)
		:address(addr), length(len) {}

};

struct mmStatus
{
	ap_uint<4>	tag;
	ap_uint<1>	interr;
	ap_uint<1>	decerr;
	ap_uint<1>	slverr;
	ap_uint<1>	okay;
};




struct ipTuple
{
	ap_uint<32>	ip_address;
	ap_uint<16>	ip_port;
};

struct openStatus
{
	ap_uint<16>	sessionID;
	bool		success;
	openStatus() {}
	openStatus(ap_uint<16> id, bool success)
		:sessionID(id), success(success) {}
};

struct appNotification
{
	ap_uint<16>			sessionID;
	ap_uint<16>			length;
	ap_uint<32>			ipAddress;
	ap_uint<16>			dstPort;
	bool				closed;
	appNotification() {}
	appNotification(ap_uint<16> id, ap_uint<16> len, ap_uint<32> addr, ap_uint<16> port)
				:sessionID(id), length(len), ipAddress(addr), dstPort(port), closed(false) {}
	appNotification(ap_uint<16> id, bool closed)
				:sessionID(id), length(0), ipAddress(0),  dstPort(0), closed(closed) {}
	appNotification(ap_uint<16> id, ap_uint<32> addr, ap_uint<16> port, bool closed)
				:sessionID(id), length(0), ipAddress(addr),  dstPort(port), closed(closed) {}
	appNotification(ap_uint<16> id, ap_uint<16> len, ap_uint<32> addr, ap_uint<16> port, bool closed)
			:sessionID(id), length(len), ipAddress(addr), dstPort(port), closed(closed) {}
};


struct appReadRequest
{
	ap_uint<16> sessionID;
	//ap_uint<16> address;
	ap_uint<16> length;
	appReadRequest() {}
	appReadRequest(ap_uint<16> id, ap_uint<16> len)
			:sessionID(id), length(len) {}
};

struct appTxMeta
{
	ap_uint<16> sessionID;
	ap_uint<32> length;
	appTxMeta() {}
	appTxMeta(ap_uint<16> id, ap_uint<16> len)
		:sessionID(id), length(len) {}
};

struct appTxRsp
{
	ap_uint<16>	sessionID;
	ap_uint<16> length;
	ap_uint<30> remaining_space;
	ap_uint<2>	error;
	appTxRsp() {}
	appTxRsp(ap_uint<16> id, ap_uint<16> len, ap_uint<30> rem_space, ap_uint<2> err)
		:sessionID(id), length(len), remaining_space(rem_space), error(err) {}
};


void mpi_reduce(	
			ap_uint<32>									mpiReduceSession,
			ap_uint<32>									mpiReduceClientID,
			ap_uint<32>									mpiReduceClientNum,
			ap_uint<32>									mpiReduceLength,

			hls::stream<ap_uint<1> >&					mpiReduceStart,
			hls::stream<ap_uint<1> >&					mpiReduceDone,

			//hbm09 interface
			hls::stream<mmCmd>&							Buffer0WriteCmd,
			hls::stream<mmCmd>&							Buffer0ReadCmd,
			hls::stream<net_axis<512> >&				Buffer0WriteData,
			hls::stream<net_axis<512> >&				Buffer0ReadData,
			// hls::stream<mmStatus>&						Buffer0WriteStatus,
			// hls::stream<mmStatus>&						Buffer0ReadStatus,			
			
			//ddr memory interface
			hls::stream<mmCmd>&							Buffer1ReadCmd,
			hls::stream<net_axis<512> >&				Buffer1ReadData,
			// hls::stream<mmStatus>&						Buffer1ReadStatus,

			//tcp tx interface
			hls::stream<appTxMeta>&						txDataReqMeta,
			hls::stream<net_axis<512> >&				txDataReq,
			// hls::stream<appTxRsp>&						txDataRsp,

			//tcp rx interface
			hls::stream<appNotification >&				rxDataRspMeta,			
			hls::stream<net_axis<512> >&				rxDataRsp);

#endif
