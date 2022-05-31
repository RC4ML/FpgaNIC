#include "app.cuh"

#include<string>
#include<errno.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include <time.h>
#include<pthread.h>
#include <unistd.h>


#include "interface.cuh"
#include "kernel.cuh"
#include "network.cuh"


using namespace std;

socket_context_t* context;
size_t single_data_length = size_t(64)*1024*1024;
int *data_gpu;
int *data_recv;
volatile int *wr;
volatile int *rd;

void *start_routine(void *arg){
	// RunConfig cfg = RunConfig();


	// sock_addr_t addr;
	// addr.ip		= cfg.next_ip_int;
	// addr.port	= 6666;

	// int* socket1;
	// cudaMalloc(&socket1,sizeof(int));

	// cudaStream_t stream_send;
	// cudaStreamCreate(&stream_send);

	// sleep(10);
	// create_socket<<<1,1,0,stream_send>>>(context,socket1);
	// connect<<<1,1,0,stream_send>>>(context,socket1,addr);
	
	// int part_index=cfg.cur_seq;
	// for(int i=0;i<cfg.total_node-1;i++){
	// 	all_reduce_wait<<<1,1,0,stream_send>>>(wr,rd);
	// 	int offset = int(part_index*single_data_length/sizeof(int));
	// 	cjprint("allreduce send, offset=%d\n",offset);
	// 	socket_send_pre<<<1,1,0,stream_send>>>(context,socket1,single_data_length);
	// 	socket_send<<<1,1024,0,stream_send>>>(context,socket1,data_gpu+offset,single_data_length);
	// 	part_index--;
	// 	if(part_index<0){
	// 		part_index+=cfg.total_node;
	// 	}
	// }
	// for(int i=0;i<cfg.total_node-1;i++){
	// 	all_reduce_wait<<<1,1,0,stream_send>>>(wr,rd);
	// 	int offset = int(part_index*single_data_length/sizeof(int));
	// 	cjprint("allreduce send, offset=%d\n",offset);
	// 	socket_send_pre<<<1,1,0,stream_send>>>(context,socket1,single_data_length);
	// 	socket_send<<<1,1024,0,stream_send>>>(context,socket1,data_gpu+offset,single_data_length);
	// 	part_index--;
	// 	if(part_index<0){
	// 		part_index+=cfg.total_node;
	// 	}
	// }
	//socket_close<<<1,1,0,stream_send>>>(context,socket1);
	cjprint("send thread done!\n");
}

void mpi_allreduce(param_test_t param_in){
	RunConfig cfg = RunConfig();
	unsigned int* buffer_addr =  ((unsigned int*)param_in.map_d_ptr);
	context = get_socket_context(buffer_addr,param_in.tlb_start_addr,param_in.controller,0);
	
	size_t total_data_length = single_data_length*cfg.total_node;
	cudaMalloc(&data_gpu,total_data_length);
	cudaMalloc(&data_recv,single_data_length);
	cudaMalloc(&wr,sizeof(int));
	cudaMalloc(&rd,sizeof(int));


	pthread_t thread;
	int ret = pthread_create(&thread,NULL,start_routine,(void*)0);
	if(ret == -1){
		printf("Create pthread error!\n");
		return;
	}
	int* socket1;
	cudaMalloc(&socket1,sizeof(int));
	cudaStream_t stream1,stream2;
	cudaEvent_t event1,event2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	connection_t* connection1;
	cudaMalloc(&connection1,sizeof(connection_t));

	all_reduce_init<<<1,1,0,stream1>>>(wr,rd);
	all_reduce_set_data<<<1,1024,0,stream1>>>(data_gpu,total_data_length,3);
	create_socket<<<1,1,0,stream1>>>(context,socket1);
	socket_listen<<<1,1,0,stream1>>>(context,socket1,6666);
	accept<<<1,1,0,stream1>>>(context,socket1,connection1);
	cudaEventRecord(event1, stream1);
	cudaStreamWaitEvent(stream2, event1,0);

	int cur_part = cfg.cur_seq-1;
	if(cur_part<0){
		cur_part+=cfg.total_node;
	}
	for(int i=0;i<cfg.total_node-1;i++){
		socket_recv<<<4,1024,0,stream1>>>(context,connection1,data_recv,single_data_length);
		socket_recv_ctrl<<<1,16,0,stream2>>>(context,connection1,data_recv,single_data_length);
		all_reduce_add<<<1,16,0,stream1>>>(data_gpu,data_recv,single_data_length,wr,cfg.total_node,cur_part);
		cudaEventRecord(event1, stream1);
		cudaStreamWaitEvent(stream2, event1,0);
		cur_part--;
		if(cur_part<0){
			cur_part+=cfg.total_node;
		}
	}

	for(int i=0;i<cfg.total_node-1;i++){
		int offset = int(cur_part*single_data_length/sizeof(int));
		cjprint("allreduce recv, offset=%d\n",offset);
		socket_recv<<<4,1024,0,stream1>>>(context,connection1,data_gpu+offset,single_data_length);
		socket_recv_ctrl<<<1,16,0,stream2>>>(context,connection1,data_gpu+offset,single_data_length);
		cudaEventRecord(event1, stream1);
		cudaStreamWaitEvent(stream2, event1,0);
		cur_part--;
		if(cur_part<0){
			cur_part+=cfg.total_node;
		}
	}
	all_reduce_verify_data<<<1,1,0,stream1>>>(data_gpu,total_data_length);
	//socket_close<<<1,1,0,stream1>>>(context,connection1);

	
}


__global__ void all_reduce_add(int * data_gpu,int * data_recv,int length_in_byte,volatile int *wr,int total_node,int cur_part){
	BEGIN_SINGLE_THREAD_DO
		cjdebug("Reduce add,cur_part:%d\n",cur_part);
	END_SINGLE_THREAD_DO
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int op_num = int(length_in_byte/(sizeof(int)));
	int iter_num = int(op_num/total_threads);
	int offset = int(cur_part*length_in_byte/sizeof(int));
	for(int i=0;i<iter_num;i++){
		data_gpu[offset+total_threads*i+index] += data_recv[total_threads*i+index];
	}
	(*wr) += 1;
	BEGIN_SINGLE_THREAD_DO
		cjdebug("reduce_add done, wr:%d\n",*wr);
	END_SINGLE_THREAD_DO
}

__global__ void all_reduce_init(volatile int * wr,volatile int * rd){
	(*wr)=1;
	(*rd)=0;
}

__global__ void all_reduce_wait(volatile int * wr,volatile int* rd){
	cjdebug("enter reduce wait, wr:%d rd:%d \n",*wr,*rd);
	while(*wr==*rd){
		//cu_sleep(1);
		//cjdebug("wr:%d\n",*rd);
	}
	(*rd)+=1;
	cjdebug("leave reduce wait, wr:%d rd:%d \n",*wr,*rd);
}

__global__ void all_reduce_set_data(int *data,size_t length_in_byte,int value){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int op_num = int(length_in_byte/(sizeof(int)));
	int iter_num = int(op_num/total_threads);
	
	for(int i=0;i<iter_num;i++){
		data[total_threads*i+index] = value;
	}
	// BEGIN_SINGLE_THREAD_DO
	// 	for(int i=0;i<op_num;i++){
	// 		data[i] = i;
	// 	}
	// END_SINGLE_THREAD_DO
}
__global__ void all_reduce_verify_data(int *data,size_t length_in_byte){
	int op_num = int(length_in_byte/(sizeof(int)));
	BEGIN_SINGLE_THREAD_DO
		int value = data[0];
		for(int i=1;i<op_num;i++){
			if(data[i]!=value){
				cjprint("verify failed,index 0 is %d but index %d is %d\n",value,i,data[i]);
				return;
			}
		}
		cjprint("verify success, data is :%d\n",value);
	END_SINGLE_THREAD_DO
}