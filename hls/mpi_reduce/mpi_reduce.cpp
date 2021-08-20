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


void cal_mpi_data(
			ap_uint<32>									mpiReduceClientID,
			ap_uint<32>									mpiReduceClientNum,
			ap_uint<32>									mpiReduceLength,

			hls::stream<ap_uint<8> >&					recvStart,
			hls::stream<ap_uint<1> >&					mpiReduceDone,
			hls::stream<ap_uint<8> >&					sendDone,
			hls::stream<ap_uint<8> >&					send_enable,

					
			//ddr memory interface
			hls::stream<mmCmd>&							Buffer0WriteCmd,
			hls::stream<mmCmd>&							Buffer1ReadCmd,
			hls::stream<net_axis<512> >&				Buffer0WriteData,
			hls::stream<net_axis<512> >&				Buffer1ReadData,
			// hls::stream<mmStatus>&						Buffer0WriteStatus,
			// hls::stream<mmStatus>&						Buffer1ReadStatus,

			hls::stream<net_axis<512> >&				sessiondata						
)
{
#pragma HLS PIPELINE II=1

	
	static net_axis<512>  	mpiReduceIn1,mpiReduceIn2,mpiReduceSum;
	



	enum cal_state {IDLE, SEND_DATA_CMD, CAL, SEND_NEXT_CMD, SEND_RESULT_CMD, SEND_RESULT, NEXT_RESULT_CMD, END};
	static cal_state cal_fsm = IDLE;


	static ap_uint<64> addr; 
	static ap_uint<32> len;
	static ap_uint<32> remain_len;
	static ap_uint<4>  recv_num;
	static ap_uint<8>  total_recv_num;
	const unsigned PKG_LENGTH = (1 << 22);

	static ap_uint<64> SessionAddr[8];

	#pragma HLS ARRAY_PARTITION variable=SessionAddr complete






	switch(cal_fsm){
	case IDLE:
		SessionAddr[0] = 0;
		SessionAddr[1] = mpiReduceLength;
		SessionAddr[2] = mpiReduceLength * 2;
		SessionAddr[3] = mpiReduceLength * 3;
		SessionAddr[4] = mpiReduceLength * 4;
		SessionAddr[5] = mpiReduceLength * 5;
		SessionAddr[6] = mpiReduceLength * 6;
		SessionAddr[7] = mpiReduceLength * 7;		
		if(!recvStart.empty()){
			recvStart.read();
			// printf("cal start------------------\n");
			total_recv_num = 0;
			if(mpiReduceClientID == 0){
				recv_num = mpiReduceClientNum;
				addr = SessionAddr[mpiReduceClientNum];
			}
			else{
				recv_num = mpiReduceClientID - 1;
				addr = SessionAddr[mpiReduceClientID - 1];
			}
			remain_len  = mpiReduceLength;
			if(mpiReduceLength > PKG_LENGTH){
				len 		= PKG_LENGTH;
			}
			else{
				len 		= mpiReduceLength;
			}
			cal_fsm = SEND_DATA_CMD;
		}
		else{
			cal_fsm =IDLE;
		}
		break;
	case SEND_DATA_CMD:		
		Buffer1ReadCmd.write(mmCmd(addr, len));
		Buffer0WriteCmd.write(mmCmd(addr, len));		
		cal_fsm = CAL;				
		break;		
	case CAL:
		if(!Buffer1ReadData.empty() && !sessiondata.empty()){
			mpiReduceIn1 = sessiondata.read();
			mpiReduceIn2 = Buffer1ReadData.read();
			mpiReduceSum.data = mpiReduceIn1.data + mpiReduceIn2.data;
			mpiReduceSum.keep = mpiReduceIn2.keep;
			mpiReduceSum.last = mpiReduceIn2.last;
			Buffer0WriteData.write(mpiReduceSum);			
			if(mpiReduceIn2.last == 1){			
				if(remain_len == 64){
					if(total_recv_num == (mpiReduceClientNum  - 1)){
						cal_fsm	= NEXT_RESULT_CMD;
					}
					else{
						cal_fsm	= SEND_NEXT_CMD;
					}					
					total_recv_num = total_recv_num + 1;
					if(recv_num == 0){
						recv_num 	= mpiReduceClientNum;
					}
					else{
						recv_num	= recv_num - 1;
					}					
				}
				else{
					cal_fsm	= SEND_DATA_CMD;
					addr = addr + PKG_LENGTH;
					if(remain_len > PKG_LENGTH){
						len = PKG_LENGTH;
					}
					else{
						len = remain_len;
					}
				}					
			}
			else{
				cal_fsm	= CAL;
			}
			remain_len = remain_len - 64;		

		}
		else{
			cal_fsm = CAL;
		}
		break;	
	case SEND_NEXT_CMD:
		send_enable.write(1);
		addr 		= SessionAddr[recv_num];
		remain_len  = mpiReduceLength;
		cal_fsm	= SEND_DATA_CMD;
		if(mpiReduceLength > PKG_LENGTH){
			len 		= PKG_LENGTH;
		}
		else{
			len 		= mpiReduceLength;
		}
		break;
	case SEND_RESULT_CMD:
		Buffer0WriteCmd.write(mmCmd(addr, len));		
		cal_fsm = SEND_RESULT;					
		break;		
	case SEND_RESULT:
		if(!sessiondata.empty()){
			mpiReduceIn1 = sessiondata.read();
			Buffer0WriteData.write(mpiReduceIn1);			
			if(mpiReduceIn1.last == 1){
				if((remain_len == 64) && (total_recv_num == (mpiReduceClientNum * 2 - 1))){
					cal_fsm	= END;
				}
				else if(remain_len == 64){
					cal_fsm	= NEXT_RESULT_CMD;
					total_recv_num = total_recv_num + 1;
					if(recv_num == 0){
						recv_num 	= mpiReduceClientNum;
					}
					else{
						recv_num	= recv_num - 1;
					}					
				}
				else{
					cal_fsm	= SEND_RESULT_CMD;
					addr = addr + PKG_LENGTH;
					if(remain_len > PKG_LENGTH){
						len = PKG_LENGTH;
					}
					else{
						len = remain_len;
					}
				}					
			}
			else{
				cal_fsm	= SEND_RESULT;
			}		
			remain_len = remain_len - 64;
		}	
		else{
			cal_fsm = SEND_RESULT;
		}
		break;	
	case NEXT_RESULT_CMD:
		send_enable.write(1);
		addr 		= SessionAddr[recv_num];
		remain_len  = mpiReduceLength;
		cal_fsm	= SEND_RESULT_CMD;
		if(mpiReduceLength > PKG_LENGTH){
			len 		= PKG_LENGTH;
		}
		else{
			len 		= mpiReduceLength;
		}
		break;		
	case END:
		if(!sendDone.empty()){
			sendDone.read();
			mpiReduceDone.write(1);
			cal_fsm = IDLE;
		}
		else{
			cal_fsm = END;
		}
	}
	

	// if(!Buffer0WriteStatus.empty()){
	// 	Buffer0WriteStatus.read();
	// }
	// if(!Buffer1ReadStatus.empty()){
	// 	Buffer1ReadStatus.read();
	// }
}

void recv_mpi_data(
			hls::stream<net_axis<512> >&				sessiondata,

			hls::stream<appNotification >&				rxDataRspMeta,
			hls::stream<net_axis<512> >&				rxDataRsp						
)
{
#pragma HLS PIPELINE II=1
	static appNotification tcpNotification;
	static net_axis<512>    tcpRecvData;
	static ap_uint<32>		c_length;

	enum recv_state {IDLE, READ_TCP_DATA};
	static recv_state recv_fsm = IDLE;	


	switch(recv_fsm){
	case IDLE:
		if(!rxDataRspMeta.empty()){
			tcpNotification = rxDataRspMeta.read();
			c_length = tcpNotification.length;
			recv_fsm = READ_TCP_DATA;
		}
		else{
			recv_fsm = IDLE;
		}
		break;	
	case READ_TCP_DATA:
		if(!rxDataRsp.empty()){
			tcpRecvData = rxDataRsp.read();
			sessiondata.write(tcpRecvData);
			if(c_length <= 64){
				c_length = 0;
				recv_fsm = IDLE;
			}
			else{
				c_length = c_length-64;
				recv_fsm = READ_TCP_DATA;
			}
		}
		else{
			recv_fsm = READ_TCP_DATA;
		}
		break;
	}


}


void send_reduce_data(
			ap_uint<32>									mpiReduceSession,
			ap_uint<32>									mpiReduceClientID,
			ap_uint<32>									mpiReduceClientNum,
			ap_uint<32>									mpiReduceLength,

			hls::stream<ap_uint<1> >&					mpiReduceStart,
			hls::stream<ap_uint<8> >&					recvStart,
			hls::stream<ap_uint<8> >&					sendDone,

			hls::stream<ap_uint<8> >&					send_enable,

			//hbm09 interface
			hls::stream<mmCmd>&							Buffer0ReadCmd,
			hls::stream<net_axis<512> >&				Buffer0ReadData,
			// hls::stream<mmStatus>&						Buffer0ReadStatus,

			hls::stream<appTxMeta>&						txDataReqMeta,
			hls::stream<net_axis<512> >&				txDataReq
			// hls::stream<appTxRsp>&						txDataRsp

)
{
#pragma HLS PIPELINE II=1
	enum send_state {IDLE, SEND_DATA_CMD, SEND_DATA, WAIT_SEND};
	static send_state send_fsm = IDLE;

	static mmCmd write_cmd;
	static ap_uint<64> addr; 
	static ap_uint<32> len;
	static ap_uint<32> remain_len;
	static ap_uint<4>  send_num;
	static ap_uint<8>  total_send_num;
	const unsigned PKG_LENGTH = (1 << 22);

	static appTxMeta mpiTxReqMate;
	static net_axis<512> sendData;

	static net_axis<512> read_data_tmp;

	static ap_uint<64> SessionAddr[8];

	static ap_uint<1> recv_start_data = 0;

	#pragma HLS ARRAY_PARTITION variable=SessionAddr complete






	switch(send_fsm){
	case IDLE:
		SessionAddr[0] = 0;
		SessionAddr[1] = mpiReduceLength;
		SessionAddr[2] = mpiReduceLength * 2;
		SessionAddr[3] = mpiReduceLength * 3;
		SessionAddr[4] = mpiReduceLength * 4;
		SessionAddr[5] = mpiReduceLength * 5;
		SessionAddr[6] = mpiReduceLength * 6;
		SessionAddr[7] = mpiReduceLength * 7;		
		if(!mpiReduceStart.empty() && !recvStart.full()){
			mpiReduceStart.read(recv_start_data);
			recvStart.write(recv_start_data);
			// printf("send start------------------\n");
			send_num	= mpiReduceClientID;
			total_send_num = 0;
			addr 		= SessionAddr[mpiReduceClientID];
			
			remain_len  = mpiReduceLength;
			send_fsm 	= SEND_DATA_CMD;
			if(mpiReduceLength > PKG_LENGTH){
				len 		= PKG_LENGTH;
			}
			else{
				len 		= mpiReduceLength;
			}
		}
		else{
			send_fsm	= IDLE;
		}
		break;
	case SEND_DATA_CMD:
		Buffer0ReadCmd.write(mmCmd(addr, len));
		mpiTxReqMate.sessionID = mpiReduceSession;
		mpiTxReqMate.length = len;
		txDataReqMeta.write(mpiTxReqMate);		
		send_fsm 	= SEND_DATA;
		break;
	case SEND_DATA:
		if(!Buffer0ReadData.empty()){
			Buffer0ReadData.read(read_data_tmp);
			txDataReq.write(read_data_tmp);				
			if(read_data_tmp.last == 1){
				if((remain_len == 64) && (total_send_num == (mpiReduceClientNum * 2 -1) )){
					send_fsm	= IDLE;
					sendDone.write(1);
				}
				else if(remain_len == 64){
					send_fsm	= WAIT_SEND;
					total_send_num = total_send_num + 1;
					if(send_num == 0){
						send_num 	= mpiReduceClientNum;
					}
					else{
						send_num	= send_num - 1;
					}					
				}
				else{
					send_fsm	= SEND_DATA_CMD;
					addr = addr + PKG_LENGTH;
					if(remain_len > PKG_LENGTH){
						len = PKG_LENGTH;
					}
					else{
						len = remain_len;
					}					
				}
					
			}
			else{
				send_fsm	= SEND_DATA;
			}
			remain_len = remain_len - 64;		
		}
		else{
			send_fsm	= SEND_DATA;
		}
		break;
	case WAIT_SEND:
		if(!send_enable.empty()){
			send_enable.read();
			addr 		= SessionAddr[send_num];
			remain_len  = mpiReduceLength;
			send_fsm	= SEND_DATA_CMD;
			if(mpiReduceLength > PKG_LENGTH){
				len 		= PKG_LENGTH;
			}
			else{
				len 		= mpiReduceLength;
			}
		}
		else{
			send_fsm	= WAIT_SEND;
		}
		break;
	}

	// if(!Buffer0ReadStatus.empty()){
	// 	Buffer0ReadStatus.read();
	// }

	// if(!txDataRsp.empty()){
	// 	txDataRsp.read();
	// }


}




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
			hls::stream<net_axis<512> >&				rxDataRsp)
{
	#pragma HLS INTERFACE axis register port=mpiReduceStart  name=s_axis_Start
	#pragma HLS INTERFACE axis register port=mpiReduceDone  name=m_axis_Done

	///////////////////////ddr interface
	#pragma HLS INTERFACE axis register port=Buffer0WriteCmd 	name=m_axis_mem0_write_cmd
	#pragma HLS INTERFACE axis register port=Buffer0WriteData 	name=m_axis_mem0_write_data
	// #pragma HLS INTERFACE axis register port=Buffer0WriteStatus name=s_axis_mem0_write_sts
	#pragma HLS INTERFACE axis register port=Buffer0ReadCmd 	name=m_axis_mem0_read_cmd
	#pragma HLS INTERFACE axis register port=Buffer0ReadData 	name=s_axis_mem0_read_data
	// #pragma HLS INTERFACE axis register port=Buffer0ReadStatus 	name=s_axis_mem0_read_sts
	#pragma HLS INTERFACE axis register port=Buffer1ReadCmd 	name=m_axis_mem1_read_cmd
	#pragma HLS INTERFACE axis register port=Buffer1ReadData 	name=s_axis_mem1_read_data
	// #pragma HLS INTERFACE axis register port=Buffer1ReadStatus 	name=s_axis_mem1_read_sts

	#pragma HLS DATA_PACK variable=Buffer0WriteCmd
	// #pragma HLS DATA_PACK variable=Buffer0WriteStatus
	#pragma HLS DATA_PACK variable=Buffer0ReadCmd
	// #pragma HLS DATA_PACK variable=Buffer0ReadStatus
	#pragma HLS DATA_PACK variable=Buffer1ReadCmd
	// #pragma HLS DATA_PACK variable=Buffer1ReadStatus

	//////////////////////tcp interface
	#pragma HLS INTERFACE axis register port=rxDataRspMeta name=s_axis_rx_data_rsp_metadata
	#pragma HLS INTERFACE axis register port=rxDataRsp name=s_axis_rx_data_rsp

	#pragma HLS INTERFACE axis register port=txDataReqMeta name=m_axis_tx_data_req_metadata
	#pragma HLS INTERFACE axis register port=txDataReq name=m_axis_tx_data_req
	// #pragma HLS INTERFACE axis register port=txDataRsp name=s_axis_tx_data_rsp
	#pragma HLS DATA_PACK variable=rxDataRspMeta
	#pragma HLS DATA_PACK variable=txDataReqMeta
	// #pragma HLS DATA_PACK variable=txDataRsp

	///////////////////////
	#pragma HLS INTERFACE ap_none register port=mpiReduceSession
	#pragma HLS INTERFACE ap_none register port=mpiReduceClientID
	#pragma HLS INTERFACE ap_none register port=mpiReduceClientNum
	#pragma HLS INTERFACE ap_none register port=mpiReduceLength
#pragma HLS DATAFLOW

	static hls::stream<ap_uint<8> > send_enable("send_enable");
	#pragma HLS stream variable=send_enable depth=32

	static hls::stream<ap_uint<8> > recvStart("recvStart");
	#pragma HLS stream variable=recvStart depth=32
	static hls::stream<ap_uint<8> > sendDone("sendDone");
	#pragma HLS stream variable=sendDone depth=32	

	static hls::stream<net_axis<512> > sessiondata("sessiondata");
	#pragma HLS stream variable=sessiondata depth=1024
	#pragma HLS DATA_PACK variable=sessiondata
	#pragma HLS RESOURCE variable=sessiondata core=RAM_2P_URAM

	// static hls::stream<net_axis<512> > ReadData0("ReadData0");
	// #pragma HLS stream variable=ReadData0 depth=1024
	// #pragma HLS DATA_PACK variable=ReadData0
	// #pragma HLS RESOURCE variable=ReadData0 core=RAM_2P_URAM
	// fifo_stream(Buffer0ReadData, ReadData0);

	// static hls::stream<net_axis<512> > ReadData1("ReadData1");
	// #pragma HLS stream variable=ReadData1 depth=1024
	// #pragma HLS DATA_PACK variable=ReadData1
	// #pragma HLS RESOURCE variable=ReadData1 core=RAM_2P_URAM
	// fifo_stream(Buffer1ReadData, ReadData1);

	// static hls::stream<net_axis<512> > TcpRxData("TcpRxData");
	// #pragma HLS stream variable=TcpRxData depth=1024
	// #pragma HLS DATA_PACK variable=TcpRxData
	// #pragma HLS RESOURCE variable=TcpRxData core=RAM_2P_URAM
	// fifo_stream(rxDataRsp, TcpRxData);

	cal_mpi_data(
					mpiReduceClientID,
					mpiReduceClientNum,
					mpiReduceLength,

					recvStart,
					mpiReduceDone,
					sendDone,
					send_enable,

					//ddr memory interface
					Buffer0WriteCmd,
					Buffer1ReadCmd,
					Buffer0WriteData,
					Buffer1ReadData,
					// Buffer0WriteStatus,
					// Buffer1ReadStatus,


					sessiondata						
	);



	recv_mpi_data(
					sessiondata,

					rxDataRspMeta,
					rxDataRsp						
	);

	send_reduce_data(
					mpiReduceSession,
					mpiReduceClientID,
					mpiReduceClientNum,
					mpiReduceLength,

					mpiReduceStart,

					recvStart,
					sendDone,
					send_enable,

					Buffer0ReadCmd,
					Buffer0ReadData,
					// Buffer0ReadStatus,

					txDataReqMeta,
					txDataReq
					// txDataRsp						
	);



}


