#include"test.hpp"
#include<iostream>
#include "cuda/vectoradd.cuh"
using namespace std;

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
	data_mover(param_mover);
	cout<<"ready for start\n";

	
	int size64 = int(50*1024*1024);
	param.controller->writeReg(40,(uint32_t)(param.addr));
	param.controller->writeReg(41,(uint32_t)((param.addr)>>32));
	param.controller->writeReg(42,0);//start_page 0-108
	param.controller->writeReg(43,200*1024*1024);//data_length  BYTE
	param.controller->writeReg(44,0);//data offset
	param.controller->writeReg(45,5);//work page size

	// param.controller->writeReg(48,(uint32_t)(param.addr+size64*1));
	// param.controller->writeReg(49,(uint32_t)(param.addr+size64*1)>>32);
	// param.controller->writeReg(50,0);//start_page 0-108
	// param.controller->writeReg(51,50*1024*1024);//data_length  BYTE
	// param.controller->writeReg(52,0);//data offset
	// param.controller->writeReg(53,25);//work page size

	// param.controller->writeReg(56,(uint32_t)(param.addr+size64*2));
	// param.controller->writeReg(57,(uint32_t)(param.addr+size64*2)>>32);
	// param.controller->writeReg(58,0);//start_page 0-108
	// param.controller->writeReg(59,50*1024*1024);//data_length  BYTE
	// param.controller->writeReg(60,0);//data offset
	// param.controller->writeReg(61,25);//work page size

	// param.controller->writeReg(64,(uint32_t)(param.addr+size64*3));
	// param.controller->writeReg(65,(uint32_t)(param.addr+size64*3)>>32);
	// param.controller->writeReg(66,0);//start_page 0-108
	// param.controller->writeReg(67,50*1024*1024);//data_length  BYTE
	// param.controller->writeReg(68,0);//data offset
	// param.controller->writeReg(69,25);//work page size

	param.controller->writeReg(46,0);
	param.controller->writeReg(46,1);
	while(1){

	}
}
void control_reg(param_test_t param){
	uint64_t start_addr  = param.controller ->getRegAddr(0);
}

void bypass_reg(param_test_t param){
	uint64_t start_addr  = param.controller ->getBypassAddr(0);
	void* addr = (void *)start_addr;
	//write_bypass(addr);
	read_bypass(addr);
	//speed = 1024*1024 * 250 * 1000000 / 1000000000 / t;
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