#include"test.hpp"
#include<iostream>
#include "cuda/interface.cuh"
#include "fpga/XDMA.h"
#include "fpga/XDMAController.h"
#include "main.h"
using namespace std;



void socket_send_test(param_test_t param_in){
	param_interface_socket_t param_out;
	param_out.tlb_start_addr = param_in.tlb_start_addr;

	param_out.ip	=	0xc0a8bd0d;
	param_out.port	=	1235;
	param_out.buffer_addr = ((unsigned int*)param_in.map_d_ptr);
	param_out.controller  = param_in.controller;
	socket_sample(param_out);
}

void socket_send_test_offload_control(param_test_t param_in){
	param_interface_socket_t param_out;
	param_out.tlb_start_addr = param_in.tlb_start_addr;

	param_out.ip	=	0xc0a8bd0d;
	param_out.port	=	1235;
	param_out.buffer_addr = ((unsigned int*)param_in.map_d_ptr);
	param_out.controller  = param_in.controller;
	socket_sample_offload_control(param_out);
}
void stream_transfer(param_test_t param){
	param_mover_t param_mover;
	param_mover.write_count_addr0	=	(void*)(param.controller->getRegAddr(572));
	param_mover.read_count_addr0	=	(void*)(param.controller->getRegAddr(47));
	param_mover.write_count_addr1	=	(void*)(param.controller->getRegAddr(573));
	param_mover.read_count_addr1	=	(void*)(param.controller->getRegAddr(55));
	param_mover.write_count_addr2	=	(void*)(param.controller->getRegAddr(574));
	param_mover.read_count_addr2	=	(void*)(param.controller->getRegAddr(63));
	param_mover.write_count_addr3	=	(void*)(param.controller->getRegAddr(575));
	param_mover.read_count_addr3	=	(void*)(param.controller->getRegAddr(71));

	int size = int(50*1024*1024/4);//50M for each channel
	param_mover.dev_addr0	=	((unsigned int*)param.map_d_ptr);//change at following stages
	param_mover.dev_addr1	=	((unsigned int*)param.map_d_ptr)+size*1;
	param_mover.dev_addr2	=	((unsigned int*)param.map_d_ptr)+size*2;
	param_mover.dev_addr3	=	((unsigned int*)param.map_d_ptr)+size*3;
	
	cout<<"ready for start\n";

	uint32_t offset				=	2;
	uint32_t data_length		=	1000*1024*1024;
	param_mover.data_length		=	data_length;
	param_mover.buffer_pages	=	1000;
	param_mover.offset			=	offset;
	param.controller->writeReg(40,(uint32_t)(param.addr));
	param.controller->writeReg(41,(uint32_t)((param.addr)>>32));
	param.controller->writeReg(42,1);//start_page 0-108
	param.controller->writeReg(43,data_length);//data_length  BYTE
	param.controller->writeReg(44,offset);//data offset
	param.controller->writeReg(45,param_mover.buffer_pages);//work page size
	
	
	
	data_mover(param_mover);

	sleep(1);

	param.controller->writeReg(46,0);
	param.controller->writeReg(46,1);
	while(1);
}

void test_throughput(param_test_t param){
	int length = 220 * 1024 * 1024;
	int offset = 0;
	int iter_num = 20;
	int mode = 1;//1 for write, 2 for read from perspective of fpga
	cout<<param.addr<<endl;
	param.controller->writeReg(32,(uint32_t)param.addr);
	param.controller->writeReg(33,(param.addr)>>32);
	for(int i=0;i<iter_num;i++){
		param.controller->writeReg(34,length);
		param.controller->writeReg(35,offset);
		param.controller->writeReg(36,0);
		param.controller->writeReg(36,mode);
		print_speed(param,length,mode);
		sleep(1);
	}
}


void print_speed(param_test_t param,int length,int mode){
	if(mode==1){
		uint32_t wr_th_sum = param.controller->readReg(566);
		cout<<"wr_th_sum:"<<wr_th_sum<<endl;
		cout<<"write speed:"<<1.0*length*250/wr_th_sum/1000<<endl;
	}else if(mode==2){
		uint32_t rd_th_sum = param.controller->readReg(567);
		cout<<"read speed:"<<1.0*length*250/rd_th_sum/1000<<endl;
	}
}