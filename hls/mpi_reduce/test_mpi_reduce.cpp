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
#include "mpi_reduce.hpp"
// #include <iostream>

int main()
{
	ap_uint<32>									mpiReduceSession = 0;
	ap_uint<32>									mpiReduceClientID = 0;
	ap_uint<32>									mpiReduceClientNum = 3;
	ap_uint<32>									mpiReduceLength = 1024;

	hls::stream<ap_uint<1> >					mpiReduceStart("mpiReduceStart");
	hls::stream<ap_uint<1> >					mpiReduceDone("mpiReduceDone");

	hls::stream<mmCmd >							Buffer0WriteCmd("Buffer0WriteCmd");
	hls::stream<net_axis<512> >					Buffer0WriteData("Buffer0WriteData");
	hls::stream<mmStatus >						Buffer0WriteStatus("Buffer0WriteStatus");

	hls::stream<mmCmd >							Buffer0ReadCmd("Buffer0ReadCmd");
	hls::stream<net_axis<512> >					Buffer0ReadData("Buffer0ReadData");
	hls::stream<mmStatus >						Buffer0ReadStatus("Buffer0ReadStatus");

	hls::stream<mmCmd >							Buffer1ReadCmd("Buffer1ReadCmd");
	hls::stream<net_axis<512> >					Buffer1ReadData("Buffer1ReadData");
	hls::stream<mmStatus >						Buffer1ReadStatus("Buffer1ReadStatus");		


	hls::stream<appTxMeta>						txDataReqMeta("txDataReqMeta");
	hls::stream<net_axis<512> >					txDataReq("txDataReq");
	hls::stream<appTxRsp>						txDataRsp("txDataRsp");


	hls::stream<appNotification>				rxDataRspMeta("rxDataRspMeta");		
	hls::stream<net_axis<512> >					rxDataRsp("rxDataRsp");
 
	mmCmd readcmd0,readcmd1;
	ap_uint<32>		readlength0=0,readlength1=0;

	net_axis<512> readData0;
	ap_uint<512>  Data0 = 0;

	net_axis<512> readData1;
	ap_uint<512>  Data1 = 0;	

	ap_uint<512>  mpiReduceData = 0;
	appTxMeta 		txcmd;
	net_axis<512> 	txdata;
	ap_uint<32> 	tx_length=0;
	appReadRequest  rxcmd;
	net_axis<512> 	rxdata;
	ap_uint<512> 	rxdatadata = 0;
 

	ap_uint<16> id = 0;
	ap_uint<16> len = 1024; 
	ap_uint<32> addr = 123456; 
	ap_uint<16> port = 1234;


	int count = 0;
	int sent_num = 0;



	while (count < 200)
	{

		if(count == 1){
			mpiReduceStart.write(1);
		}

		if(!Buffer0WriteCmd.empty()){
			Buffer0WriteCmd.read();	
		}

		if(!Buffer0WriteData.empty()){
			Buffer0WriteData.read();	
		}

		if(!Buffer0ReadCmd.empty() && (readlength0 == 0)){
			Buffer0ReadCmd.read(readcmd0);
			readlength0 = readcmd0.length;
			std::cout <<  "readcmd0 addr: " << readcmd0.address << " length: " << readcmd0.length << std::endl;
		}

		if(!Buffer0ReadData.full() && (readlength0 != 0)){
			readData0.keep = 0xFFFFFFFFFFFFFFFF;
			readData0.data = Data0;
			if(readlength0 == 64){
				readData0.last = 1;
			}
			else{
				readData0.last = 0;
			}
			Buffer0ReadData.write(readData0);
			// std::cout <<  "Data0: " << Data0 << std::endl;
			readlength0 = readlength0 - 64;
			Data0++;
			
		}

		if(!Buffer1ReadCmd.empty() && (readlength1 == 0)){
			Buffer1ReadCmd.read(readcmd1);
			readlength1 = readcmd1.length;
			std::cout <<  "readcmd1 addr: " << readcmd1.address << " length: " << readcmd1.length << std::endl;
		}

		if(!Buffer1ReadData.full() && (readlength1 != 0)){
			readData1.keep = 0xFFFFFFFFFFFFFFFF;
			readData1.data = Data1;
			if(readlength1 == 64){
				readData1.last = 1;
			}
			else{
				readData1.last = 0;
			}
			Buffer1ReadData.write(readData1);
			// std::cout <<  "Data1: " << Data0 << std::endl;
			readlength1 = readlength1 - 64;
			Data1++;
			
		}


		if(!txDataReqMeta.empty() && (tx_length == 0)){
			txDataReqMeta.read(txcmd);
			tx_length = txcmd.length;
			std::cout <<  " tx_length: " << txcmd.length << std::endl;
		}

		if(!txDataReq.empty() && (tx_length != 0)){
			txDataReq.read(txdata);
			tx_length = tx_length - 64;
		}



		if(!rxDataRspMeta.full()){
			id = mpiReduceSession;
			rxDataRspMeta.write(appNotification(id, len, addr, port));
			// std::cout << std::hex << "rx sessionID: " << rxcmd.sessionID << ", len: " << rxcmd.length << std::endl;
		}

		if(!rxDataRsp.full() ){
			rxdata.data = rxdatadata;
			rxdata.keep = 0xFFFFFFFFFFFFFFFF;
			rxDataRsp.write(rxdata);
			rxdatadata ++;
			if(rxdatadata == 16){
				rxdatadata = 0;
			}
		}

		if(!mpiReduceDone.empty()){
			std::cout <<  " mpiReduceDone: " << mpiReduceDone.read() << std::endl;
		}



 mpi_reduce(	
					mpiReduceSession,
					mpiReduceClientID,
					mpiReduceClientNum,
					mpiReduceLength,

					mpiReduceStart,
					mpiReduceDone,
			//hbm09 interface
					Buffer0WriteCmd,
					Buffer0ReadCmd,
					Buffer0WriteData,
					Buffer0ReadData,
					Buffer0WriteStatus,
					Buffer0ReadStatus,			
			
			//ddr memory interface
					Buffer1ReadCmd,
					Buffer1ReadData,
					Buffer1ReadStatus,

					txDataReqMeta,
					txDataReq,
					txDataRsp,

					rxDataRspMeta,			
					rxDataRsp);



		count++;

	}

	return 0;
}
