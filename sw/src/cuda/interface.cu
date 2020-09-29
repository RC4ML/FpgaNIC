#include "interface.cuh"
#include "kernel.cuh"
#include "network.cuh"
 


void socket_sample(param_interface_socket_t param_in){
	socket_context_t* context = get_socket_context(param_in.buffer_addr,param_in.tlb_start_addr,param_in.controller);
	int user_blocks=4;
	int user_thread_num=256;
	dim3 threads_per_block(user_thread_num,1);	
	dim3 num_blocks(user_blocks,1);	  
	int * data;
	size_t length = 4*256*1024*1024;
	cudaMalloc(&data,length);

	sock_addr_t addr;
	addr.ip = param_in.ip;
	addr.mac = param_in.mac;
	addr.port = param_in.port;

	int* socket1;
	cudaMalloc(&socket1,sizeof(int));
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	create_socket<<<1,1,0,stream>>>(context,socket1);
	compute<<<1,1024,0,stream>>>(data,length,3);
	socket_send<<<1,2,0,stream>>>(context,socket1,data,1024,addr);
	// sleep(10);
	// printf("%x\n",param_in.controller.readReg());
	// printf("%x\n",param_in.controller.readReg());
	// printf("%x\n",param_in.controller.readReg());
	// printf("%x\n",param_in.controller.readReg());
	// printf("%x\n",param_in.controller.readReg());
	// printf("%x\n",param_in.controller.readReg());
	// printf("%x\n",param_in.controller.readReg());
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
	
	writeBypassReg<<<numBlocks,threadsPerBlock>>>(devPtrAddr,device_blocks);
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
	
	readBypassReg<<<numBlocks,threadsPerBlock>>>(devPtrAddr,device_blocks);
	cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}


