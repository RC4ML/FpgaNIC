#include"test.hpp"
#include<iostream>
#include "cuda/vectoradd.cuh"
using namespace std;

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