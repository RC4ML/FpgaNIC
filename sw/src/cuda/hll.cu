#include"hll.cuh"
#include "kernel.cuh"
#include "network.cuh"
#include "util.cuh"
#include "tool/log.hpp"
using namespace std;
__global__ void init(socket_context_t* context,int *mem){
	for(int i=0;i<65536;i++){
		mem[i]=0;
	}
	context->hll_mem = mem;
}
void hll_sample(param_test_t param_in){
	// {
	// 	int *data;
	// 	cudaMalloc(&data,sizeof(int)*1024*1024);
	// 	int *mem;
	// 	cudaMalloc(&mem,sizeof(int)*65536);
	// 	hll_test<<<1,16>>>(data,mem);
	// 	while(1);
	// }
	unsigned int* buffer_addr =  ((unsigned int*)param_in.map_d_ptr);
	socket_context_t* context = get_socket_context(buffer_addr,param_in.tlb_start_addr,param_in.controller,app_type);
	
	sock_addr_t addr;//6 -> 4
	addr.ip = 0xc0a8bd0a;//0a => amax4
	addr.port = 6666;

	int* socket1;
	cudaMalloc(&socket1,sizeof(int));
	connection_t* connection1;
	cudaMalloc(&connection1,sizeof(connection_t));
	
	cudaStream_t stream1,stream2;
	cudaEvent_t event1,event2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);
	sleep(1);

	cjprint("start user code:\n");
	int verify_data_offset = 5;
	size_t transfer_length = size_t(50)*1024*1024*4;
	param_in.controller->writeReg(165,(unsigned int)(transfer_length/64));//count code
	param_in.controller->writeReg(183,(unsigned int)(transfer_length/64));
	if(app_type==0){
		sleep(3);
		int * data;
		cudaMalloc(&data,transfer_length);
		compute<<<1,1024,0,stream1>>>(data,transfer_length,verify_data_offset);

		create_socket<<<1,1,0,stream1>>>(context,socket1);
		connect<<<1,1,0,stream1>>>(context,socket1,addr);
		socket_send_pre<<<1,1,0,stream1>>>(context,socket1,transfer_length);
		socket_send<<<2,1024,0,stream1>>>(context,socket1,data,transfer_length);
	}else if(app_type==1){
		int *mem;
		cudaMalloc(&mem,sizeof(int)*65536);
		init<<<1,1>>>(context,mem);

		int * data_recv;
		cudaMalloc(&data_recv,transfer_length);
		create_socket<<<1,1,0,stream1>>>(context,socket1);
		socket_listen<<<1,1,0,stream1>>>(context,socket1,6666);
		accept<<<1,1,0,stream1>>>(context,socket1,connection1);
		cudaEventRecord(event1, stream1);
		cudaStreamWaitEvent(stream2, event1,0);
		socket_recv<<<4,1024,0,stream1>>>(context,connection1,(int *)data_recv,transfer_length);
		socket_recv_ctrl<<<1,16,0,stream2>>>(context,connection1,(int *)data_recv,transfer_length);
		verify<<<1,1,0,stream1>>>(data_recv,transfer_length,verify_data_offset);
	}

}
