#ifndef APP_CUH
#define APP_CUH

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include <string>
#include <sys/ioctl.h>
#include <arpa/inet.h>

#include "main.h"
#include "tool/log.hpp"

#define MAX_MACHINE 9
//from 0
using namespace std;

class RunConfig{
	public:
		int total_node;
		string ip_tbl[MAX_MACHINE];
		bool valid_tbl[MAX_MACHINE];
		int next_index;
		int cur_index;
		int cur_seq;
		int next_seq;
		unsigned int next_ip_int;

		unsigned int get_ip_int(string ip){
			return (unsigned int)htonl(inet_addr(ip.data()));
		}

		RunConfig(){
			string prefix="192.168.189.";
			for(int i=0;i<MAX_MACHINE;i++){
				ip_tbl[i] = prefix+to_string(i+6);
				valid_tbl[i]=0;
			}

			valid_tbl[3]=1;
			valid_tbl[4]=1;
			//valid_tbl[6]=1;

			total_node = accumulate(valid_tbl,valid_tbl+MAX_MACHINE,0);
			if(valid_tbl[node_index] == false){
				cjerror("ERROR: node %d not valid\n",node_index);
			}

			{
				int start=node_index+1;
				for(int i=0;i<MAX_MACHINE;i++){
					if(valid_tbl[start%MAX_MACHINE]){
						next_index = start%MAX_MACHINE;
					}else{
						start++;
					}
				}
			}
			cur_index = node_index;
			{
				int count=0;
				for(int i=0;i<MAX_MACHINE;i++){
					if(i==cur_index){
						cur_seq = count;
					}
					if(i==next_index){
						next_seq = count;
					}
					if(valid_tbl[i]){
						count++;
					}
				}
			}
			next_ip_int = get_ip_int(ip_tbl[next_index]);
			cjprint("total node:%d, cur_index:%d next_index:%d cur_seq:%d next_seq:%d\n",total_node,cur_index,next_index,cur_seq,next_seq);
			cjprint("cur ip:%s %x next ip:%s %x\n",ip_tbl[cur_index].c_str(),get_ip_int(ip_tbl[cur_index]),ip_tbl[next_index].c_str(),get_ip_int(ip_tbl[next_index]));
		}

};

void mpi_allreduce(param_test_t param_in);
__global__ void all_reduce_add(int * data_gpu,int * data_recv,int length_in_byte,volatile int *wr,int total_node,int cur_part);
__global__ void all_reduce_init(volatile int * wr,volatile int * rd);
__global__ void all_reduce_wait(volatile int * wr,volatile int* rd);
__global__ void all_reduce_set_data(int *data,size_t length_in_byte,int value);
__global__ void all_reduce_verify_data(int *data,size_t length_in_byte);
#endif