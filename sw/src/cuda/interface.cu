#include "interface.cuh"
#include "kernel.cuh"
#include "network.cuh"
#include "inference_util.cuh"
#include "sys/time.h"
#include <fstream>
#include <iostream>
#include "tool/log.hpp"
#include "main.h"

using namespace std;

void socket_sample_offload_control(param_interface_socket_t param_in){
	socket_context_t* context = get_socket_context(param_in.buffer_addr,param_in.tlb_start_addr,param_in.controller,0);
	int * data;
	size_t total_data_length = size_t(1)*1024*1024*1024;
	cudaMalloc(&data,total_data_length);
	sock_addr_t addr;
	addr.ip = param_in.ip;
	addr.port = param_in.port;

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
	int verify_data_offset = 5;
	size_t transfer_data_length = transfer_megabyte*1024*1024;
	param_in.controller->writeReg(165,(unsigned int)(transfer_data_length/64));//count code
	param_in.controller->writeReg(183,(unsigned int)(transfer_data_length/64));
	if(app_type==0){
		sleep(3);
		create_socket<<<1,1,0,stream1>>>(context,socket1);
		compute<<<1,1024,0,stream1>>>(data,total_data_length,verify_data_offset);
		connect<<<1,1,0,stream1>>>(context,socket1,addr);
		size_t max_block_size = max_block_size_kilobyte * 1024;
		socket_send_pre<<<1,1,0,stream1>>>(context,socket1,transfer_data_length,max_block_size);
		
		unsigned int* ctrl_data;
		cudaMallocManaged(&ctrl_data, sizeof(unsigned int)*16*4);
		for(int i=0;i<transfer_data_length/max_block_size/4;i++){
			// printf("start send kernel\n");
			socket_send_offload_control<<<4,1024,0,stream1>>>(context,socket1,data,transfer_data_length,i*4,ctrl_data);
			cudaEventRecord(event1, stream1);
			cudaEventSynchronize(event1);
			for(int j=0;j<4;j++){
				param_in.controller->writeBypassReg(3,(uint64_t*)(ctrl_data+16*j));
			}
		}
		printf("write data done\n");
	}else if(app_type==1){
		//server code
		create_socket<<<1,1,0,stream1>>>(context,socket1);
		socket_listen<<<1,1,0,stream1>>>(context,socket1,1235);
		accept<<<1,1,0,stream1>>>(context,socket1,connection1);
		cudaEventRecord(event1, stream1);
		cudaStreamWaitEvent(stream2, event1,0);
		socket_recv<<<4,1024,0,stream1>>>(context,connection1,data,transfer_data_length);
		socket_recv_ctrl<<<1,16,0,stream2>>>(context,connection1,data,transfer_data_length);
		cudaEventRecord(event2, stream2);
		cudaStreamWaitEvent(stream1, event2,0);
		verify<<<1,1,0,stream1>>>(data,transfer_data_length,verify_data_offset);
	}else{
		cjerror("app_type not set!\n");
	}
	sleep(5);
	cudaError_t cudaerr = cudaPeekAtLastError();
	ErrCheck(cudaerr);
}


void socket_sample(param_interface_socket_t param_in){
	socket_context_t* context = get_socket_context(param_in.buffer_addr,param_in.tlb_start_addr,param_in.controller,0);

	int * data;
	size_t total_data_length = size_t(1)*1024*1024*1024;
	cudaMalloc(&data,total_data_length);

	sock_addr_t addr;
	addr.ip = param_in.ip;
	addr.port = param_in.port;

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
	size_t transfer_data_length = transfer_megabyte*1024*1024;
	param_in.controller->writeReg(165,(unsigned int)(transfer_data_length/64));//count code
	param_in.controller->writeReg(183,(unsigned int)(transfer_data_length/64));
	size_t max_block_size = max_block_size_kilobyte * 1024;
	if(app_type==0){
		//client code
		create_socket<<<1,1,0,stream1>>>(context,socket1);
		compute<<<1,1024,0,stream1>>>(data,total_data_length,verify_data_offset);
		connect<<<1,1,0,stream1>>>(context,socket1,addr);
		socket_send_pre<<<1,1,0,stream1>>>(context,socket1,transfer_data_length,max_block_size);
		socket_send<<<4,1024,0,stream1>>>(context,socket1,data,transfer_data_length);
		//socket_send<<<1,1024,0,stream1>>>(context,socket1,data,transfer_data_length/2);
		//socket_send<<<1,1024,0,stream1>>>(context,socket1,data+(transfer_data_length/2/4),transfer_data_length/2);
		//socket_close<<<1,1,0,stream>>>(context,socket1);
	}else if(app_type==1){
		//server code
		create_socket<<<1,1,0,stream1>>>(context,socket1);
		socket_listen<<<1,1,0,stream1>>>(context,socket1,1235);
		accept<<<1,1,0,stream1>>>(context,socket1,connection1);
		cudaEventRecord(event1, stream1);
		cudaStreamWaitEvent(stream2, event1,0);
		socket_recv<<<4,1024,0,stream1>>>(context,connection1,data,transfer_data_length);
		socket_recv_ctrl<<<1,16,0,stream2>>>(context,connection1,data,transfer_data_length);
		// cudaError_t cudaerr = cudaDeviceSynchronize();
		// ErrCheck(cudaerr);
		cudaEventRecord(event2, stream2);
		cudaStreamWaitEvent(stream1, event2,0);
		verify<<<1,1,0,stream1>>>(data,transfer_data_length,verify_data_offset);

		// cudaError_t cudaerr = cudaDeviceSynchronize();
		// ErrCheck(cudaerr);
		// sleep(10);

		//socket_close<<<1,1,0,stream>>>(context,connection1);
	}else{
		cjerror("app_type not set!\n");
	}
	sleep(5);
	cudaError_t cudaerr = cudaPeekAtLastError();
	ErrCheck(cudaerr);


}




void data_mover(param_mover param_mover){
	
	cudaError_t err = cudaHostRegister(param_mover.write_count_addr0,4,cudaHostRegisterIoMemory);
	ErrCheck(err);
	err = cudaHostRegister(param_mover.read_count_addr0,4,cudaHostRegisterIoMemory);
	ErrCheck(err);

	err = cudaHostRegister(param_mover.write_count_addr1,4,cudaHostRegisterIoMemory);
	ErrCheck(err);
	err = cudaHostRegister(param_mover.read_count_addr1,4,cudaHostRegisterIoMemory);
	ErrCheck(err);

	err = cudaHostRegister(param_mover.write_count_addr2,4,cudaHostRegisterIoMemory);
	ErrCheck(err);
	err = cudaHostRegister(param_mover.read_count_addr2,4,cudaHostRegisterIoMemory);
	ErrCheck(err);

	err = cudaHostRegister(param_mover.write_count_addr3,4,cudaHostRegisterIoMemory);
	ErrCheck(err);
	err = cudaHostRegister(param_mover.read_count_addr3,4,cudaHostRegisterIoMemory);
	ErrCheck(err);

	param_cuda_thread_t param_cuda;

	param_cuda.data_length 		=	param_mover.data_length;
	param_cuda.offset			=	param_mover.offset;
	param_cuda.buffer_pages		=	param_mover.buffer_pages;

	cudaHostGetDevicePointer((void **) &(param_cuda.devReadCountAddr0), param_mover.read_count_addr0, 0);
	cudaHostGetDevicePointer((void **) &(param_cuda.devReadCountAddr1), param_mover.read_count_addr1, 0);
	cudaHostGetDevicePointer((void **) &(param_cuda.devReadCountAddr2), param_mover.read_count_addr2, 0);
	cudaHostGetDevicePointer((void **) &(param_cuda.devReadCountAddr3), param_mover.read_count_addr3, 0);

	cudaHostGetDevicePointer((void **) &(param_cuda.devWriteCountAddr0), param_mover.write_count_addr0, 0);
	cudaHostGetDevicePointer((void **) &(param_cuda.devWriteCountAddr1), param_mover.write_count_addr1, 0);
	cudaHostGetDevicePointer((void **) &(param_cuda.devWriteCountAddr2), param_mover.write_count_addr2, 0);
	cudaHostGetDevicePointer((void **) &(param_cuda.devWriteCountAddr3), param_mover.write_count_addr3, 0);

	param_cuda.devVAddr0 = param_mover.dev_addr0;
	param_cuda.devVAddr1 = param_mover.dev_addr1;
	param_cuda.devVAddr2 = param_mover.dev_addr2;
	param_cuda.devVAddr3 = param_mover.dev_addr3;

	int hostBlocks=1;
	int threadNum=1024;//8 128
	param_cuda.threadsPerBlock = threadNum;
	param_cuda.blocks = hostBlocks;
	
	
	dim3 threadsPerBlock(threadNum,1);	
	dim3 numBlocks(hostBlocks,1);	  

	//local copy
	// unsigned int * src;
	// cudaMalloc(&(src),2000*1024*1024);
	// param_cuda.devVAddr0 = src;

	cudaMalloc(&(param_cuda.dstAddr0),param_mover.data_length);
	// cudaMalloc(&(param_cuda.dstAddr1),200*1024*1024);
	// cudaMalloc(&(param_cuda.dstAddr2),200*1024*1024);
	// cudaMalloc(&(param_cuda.dstAddr3),200*1024*1024);
	movThread<<<numBlocks,threadsPerBlock>>>(param_cuda); 
	// cudaError_t cudaerr = cudaDeviceSynchronize();
	// ErrCheck(cudaerr);
}

void write_bypass(void* addr){
	int peak_clk = 1;
	int device = 0;
	cudaError_t e = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device);
	printf("%d\n",peak_clk);
	cudaError_t err = cudaHostRegister((void *)addr,1024*1024,cudaHostRegisterIoMemory);
	if (err != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(err));
	volatile unsigned int *devPtrAddr;
	cudaHostGetDevicePointer((void **) &devPtrAddr, (void *) addr, 0);
	int host_blocks=1;
	dim3 threadsPerBlock(16,1);
	int *device_blocks;
	cudaMalloc(&device_blocks, 1 * sizeof(int));
	cudaMemcpy(device_blocks,&host_blocks,sizeof(int)*1,cudaMemcpyHostToDevice);
	dim3 numBlocks(host_blocks,1);
	
	//writeBypassReg<<<numBlocks,threadsPerBlock>>>(devPtrAddr,device_blocks);
	cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}
void read_bypass(void* addr){
	int peak_clk = 1;
	int device = 0;
	cudaError_t e = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device);
	printf("%d\n",peak_clk);
	cudaError_t err = cudaHostRegister((void *)addr,1024*1024,cudaHostRegisterIoMemory);
	if (err != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(err));
	volatile unsigned int *devPtrAddr;
	cudaHostGetDevicePointer((void **) &devPtrAddr, (void *) addr, 0);
	int host_blocks=1;
	dim3 threadsPerBlock(16,1);
	int *device_blocks;
	cudaMalloc(&device_blocks, 1 * sizeof(int));
	cudaMemcpy(device_blocks,&host_blocks,sizeof(int)*1,cudaMemcpyHostToDevice);
	dim3 numBlocks(host_blocks,1);
	
	//readBypassReg<<<numBlocks,threadsPerBlock>>>(devPtrAddr,device_blocks);
	cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}


