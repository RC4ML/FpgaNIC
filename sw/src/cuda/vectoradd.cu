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
	int sum=0;
	clock_t s = clock64();
	sum+=dev_addr[index];
	__threadfence();
	
	clock_t e = clock64();
	printf("latency:%lu\n",e-s);
	printf("%d \n",sum);
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