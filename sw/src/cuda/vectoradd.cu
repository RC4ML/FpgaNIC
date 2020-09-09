#include "vectoradd.cuh"
__global__ void VecAdd(float** A, float** C){
	// printf("%d %d %d %d\n", blockIdx.x,blockIdx.y, threadIdx.x,threadIdx.y);
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	// C[i][j] = A[i][j]*A[i][j];
	// printf("%d %d %f\n",i,j,C[i][j]);
	long i=1;
	while(i++){
		if(i==0x1000000000){
			i=1;
			printf("gpu run\n");
		}
	}
}
__global__ void movThread(param_mov_thread_t param){
	//printf("readCount:%d 	writeCount:%d\n",devReadCountAddr[0],devWriteCountAddr[0]);
	int rCount[128];
	unsigned int* dataAddr = param.devVAddr0 + int(2*1024*1024/4);
	int step = int(2*1024*1024/4);
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	// if(index==0){
	// 	printf("%llx %llx\n",param.devVAddr0,param.devVAddr0+1);
	// }
	if(0 == index%(blockDim.x)){
		rCount[blockIdx.x]=param.devReadCountAddr0[0];
	}
	if(0 == index){
		printf("index:%d rcount:%d devReadCountAddr0: %d   devWriteCountAddr: %d\n",index,rCount[0],param.devReadCountAddr0[0],param.devWriteCountAddr0[0]);
	}
	int flag=0;

	//todo
	// while(1){
	// 	if(0==index){
	// 		if(flag==0){
	// 			flag=1;
	// 			printf("%d %d\n",index,flag);
	// 		}
	// 	}
	// }
	while(1){
		// if(index==0){
		// 	param.devReadCountAddr0[0] =param.devWriteCountAddr0[0];
		// }
		
		while(index%(blockDim.x)==0 && rCount[blockIdx.x]==param.devWriteCountAddr0[0]){
			continue;
		}
		__syncthreads();
		unsigned int* startDstAddr = param.dstAddr0+step*(rCount[blockIdx.x]%100);
		unsigned int* startSrcAddr = dataAddr + step*(rCount[blockIdx.x]%100);
		int length = int(step/param.threadsPerBlock/param.blocks);
		int stride = param.threadsPerBlock*param.blocks;
		for(int i=0;i<length;i++){
			startDstAddr[i*stride+index] = startSrcAddr[i*stride+index];
		}
		if(index%(blockDim.x)==0){
			rCount[blockIdx.x]+=1;
		}
		__syncthreads();
		if(index==0){
			param.devReadCountAddr0[0] = rCount[0];
			//printf("index:%d  devReadCountAddr:%d   devWriteCountAddr:%d\n",index,param.devReadCountAddr0[0],param.devWriteCountAddr0[0]);
		}
	}
	// clock_t s,e;
	// int length = int(step/param.threadsPerBlock/param.blocks);
	// int stride = param.threadsPerBlock*param.blocks;
	// s = clock64();
	// for(int pageId = 0;pageId<1000;pageId++){
	// 	unsigned int* startDstAddr = param.dstAddr0+step*(pageId%100);
	// 	unsigned int* startSrcAddr = dataAddr + step*(pageId%100);
	// 	for(int i=0;i<length;i++){
	// 		startDstAddr[i*stride+index] = startSrcAddr[i*stride+index];
	// 	}
	// }
	// __threadfence();
	// e = clock64();
	// printf("latency:%lu  %d  %d  %d  %d\n",e-s,length,step,param.threadsPerBlock,param.blocks);

}
__global__ void writeBypassReg(volatile unsigned int *dev_addr,int *blocks){
	//printf("enter writeBypassReg thread with mapped dev_addr:%x\n",dev_addr);
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int stride = blocks[0]*blockDim.x;
	int num = int(1024*1024/4/stride);
	int sum=0;
	for(int i=0;i<num;i++){
		dev_addr[i*stride+index]=i;
		sum+=dev_addr[i*stride+index];
	}
	printf("%d %d %d \n",index,stride,num);
	printf("%d \n",sum);
}
__global__ void readBypassReg(volatile unsigned int *dev_addr,int *blocks){
	//printf("enter readBypassReg thread with mapped dev_addr:%x\n",dev_addr);
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int stride = blocks[0]*blockDim.x;
	int next_addr,sum;
	
	next_addr=dev_addr[1600+index];
	clock_t s = clock64();
	sum = dev_addr[next_addr+index];
	// __threadfence();
	clock_t e = clock64();
	printf("latency:%lu\n",e-s);
	printf("%d %d \n",next_addr,sum);
}
__global__ void writeReg(volatile unsigned int *dev_addr,int *blocks){
	
	printf("enter writeReg thread with addr:%x\n",dev_addr);
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int stride = blocks[0]*blockDim.x;
	int num = int(1024*1024/4/stride);
	clock_t start_clock = clock();
	// #pragma unroll 1
	int sum=0;
	for(int i=0;i<num;i++){
		dev_addr[i*stride+index]=i;
		//__syncthreads();
		//__threadfence();
		//__threadfence_block();
		//__threadfence_system();
		sum+=dev_addr[i*stride+index];
	}

	printf("%d %d %d \n",index,stride,num);
	printf("%d \n",sum);
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

	param_mov_thread_t param_cuda;

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

	int hostBlocks=8;
	int threadNum=128;
	param_cuda.threadsPerBlock = threadNum;
	param_cuda.blocks = hostBlocks;
	
	dim3 threadsPerBlock(threadNum,1);	
	dim3 numBlocks(hostBlocks,1);	  

	//local copy
	// unsigned int * src;
	// cudaMalloc(&(src),2000*1024*1024);
	// param_cuda.devVAddr0 = src;

	cudaMalloc(&(param_cuda.dstAddr0),2000*1024*1024);
	cudaMalloc(&(param_cuda.dstAddr1),200*1024*1024);
	cudaMalloc(&(param_cuda.dstAddr2),200*1024*1024);
	cudaMalloc(&(param_cuda.dstAddr3),200*1024*1024);
	movThread<<<numBlocks,threadsPerBlock>>>(param_cuda); 
	// volatile unsigned int *devReadCountAddr0;
	// volatile unsigned int *devWriteCountAddr0;
	// cudaHostGetDevicePointer((void **) &devReadCountAddr, read_count_addr, 0);
	// cudaHostGetDevicePointer((void **) &devWriteCountAddr, write_count_addr, 0);
	// int host_blocks=16;
	// int threadNum=256;
	// // dim3 threadsPerBlock(16,1);
	// dim3 threadsPerBlock(threadNum,1);
	// int *device_blocks;
	// cudaMalloc(&device_blocks, 1 * sizeof(int));
	// cudaMemcpy(device_blocks,&host_blocks,sizeof(int)*1,cudaMemcpyHostToDevice);
	// dim3 numBlocks(host_blocks,1);
	
	// unsigned int * dstAddr;
	// cudaMalloc(&dstAddr,2000*1024*1024);
	// movThread<<<numBlocks,threadsPerBlock>>>(devReadCountAddr,devWriteCountAddr,(unsigned int *)v_addr,dstAddr,threadNum);
	// cudaError_t cudaerr = cudaDeviceSynchronize();
    // if (cudaerr != cudaSuccess)
    //     printf("kernel launch failed with error \"%s\".\n",
    //            cudaGetErrorString(cudaerr));
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



void useCUDA(){
	int row = 1;
	int col = 1;
	float** d_a;
	float** d_c;
	float** h_a;
	float** h_c;
	float* d_a_data;
	float* d_c_data;
	float* h_a_data;
	h_a = (float **)malloc(sizeof(float*)*row);
	h_c = (float **)malloc(sizeof(float*)*row);
	h_a_data = (float *)malloc(sizeof(float)*col*row);
	cudaMalloc((void **)&d_a,sizeof(float*)*row);
	cudaMalloc((void **)&d_c,sizeof(float*)*row);
	cudaMalloc((void **)&d_a_data,sizeof(float)*row*col);
	cudaMalloc((void **)&d_c_data,sizeof(float)*row*col);
	for(int i=0;i<row*col;i++){
		h_a_data[i] = i;
	}
	cudaMemcpy(d_a_data,h_a_data,sizeof(float)*row*col,cudaMemcpyHostToDevice);
	for(int i=0;i<row;i++){
		h_a[i] = d_a_data+col*i;
		h_c[i] = d_c_data+col*i;
	}
	cudaMemcpy(d_a,h_a,sizeof(float*)*row,cudaMemcpyHostToDevice);
	cudaMemcpy(d_c,h_c,sizeof(float*)*row,cudaMemcpyHostToDevice);
	int row_,col_;
	row_ = 1;
	col_ = 1;
	dim3 threadsPerBlock(row_,col_);
	dim3 numBlocks(row/threadsPerBlock.x,col/threadsPerBlock.y);
	VecAdd<<<numBlocks,threadsPerBlock>>>(d_a, d_c);
	// cudaError_t cudaerr = cudaDeviceSynchronize();
    // if (cudaerr != cudaSuccess)
    //     printf("kernel launch failed with error \"%s\".\n",
    //            cudaGetErrorString(cudaerr));
}