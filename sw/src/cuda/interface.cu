#include "interface.cuh"
#include "kernel.cuh"
#include "network.cuh"
#include "sys/time.h"
#include <fstream>
#include <iostream>

using namespace std;
ofstream outfile;
__global__ void GlobalCopy(int *out, const int *in, size_t N )
{
    int temp[16];
	N=size_t(N/4);
	//avoid accessing cache, assure cold-cache access
	int start = (blockIdx.x * blockDim.x + threadIdx.x);
    int step = (blockDim.x * gridDim.x);
    // int step = 16 ;

    // printf("start:%d\n",step);
	int i;

    for ( i = start; i < N; i += step*1 ) {
        for ( int j = 0; j <1; j++ ) {
            // int index = i;//+j*blockDim.x;;
            temp[j] += in[i + j*step];
        }
    }
    for(int j=0;j<1;j++){
        out[j] = temp[j];
    }
    
}



__global__ void gpu_pressure(volatile unsigned int *data_addr,int iter,size_t block_length,unsigned int *out){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	size_t op_num = size_t(block_length/sizeof(int));
	int total_threads = blockDim.x*gridDim.x;
	size_t iter_num = size_t(op_num/total_threads);
	clock_t s,e;

	for(size_t i=0;i<iter_num;i++){
		data_addr[total_threads*i+index] = 1;
	}
	out[index]=0;
	BEGIN_SINGLE_THREAD_DO
		printf("###########gpu move start! total_threads:%d  opnum:%ld  iter_num:%d\n",total_threads,op_num,iter_num);
		s = clock64();
	END_SINGLE_THREAD_DO

	
	for(int it=0;it<iter;it++){
		for(size_t i=0;i<iter_num;i++){
			out[index]+=data_addr[total_threads*i+index];
		}
	}

	
	BEGIN_SINGLE_THREAD_DO
		e = clock64();
		float time = (e-s)/1.41/1e9;
		float speed = 1.0*block_length*iter/1024/1024/1024/time;
		// printf("e-s:%ld  time=%f speed=%f GB/s \n",e-s,time,speed);
	END_SINGLE_THREAD_DO
}

void gpu_benchmark(param_test_t param_in,int burst,int ops,int start){
	sleep(1);
	fpga::XDMAController* controller = param_in.controller;

	unsigned long tlb_start_addr_value = (unsigned long)param_in.tlb_start_addr;
	controller->writeReg(32,(unsigned int)tlb_start_addr_value);
	controller->writeReg(33,(unsigned int)(tlb_start_addr_value>>32));

	unsigned int* recv_tlb_start_addr = param_in.tlb_start_addr+int((100*1024*1024)/sizeof(int));
	unsigned long recv_tlb_start_addr_value = (unsigned long)recv_tlb_start_addr;
	controller->writeReg(34,(unsigned int)recv_tlb_start_addr_value);
	controller->writeReg(35,(unsigned int)(recv_tlb_start_addr_value>>32));

	cout<<"start fpga workload\n";
	int rd_sum,wr_sum;
	float rd_speed,wr_speed;
	int total_length = 100*1024*1024 ;

	controller ->writeReg(36,total_length);
	controller ->writeReg(37,ops);
	controller ->writeReg(38,burst);
	controller ->writeReg(39,0);
	controller ->writeReg(39,start);
	sleep(10);
  	controller ->writeReg(39,0);
	
	
	rd_sum = controller ->readReg(577);
	wr_sum = controller ->readReg(576);
	// cout << "wr_sum: " << wr_sum <<endl; 
  	// cout << "rd_sum: " << rd_sum <<endl;
	wr_speed = 1.0*burst*ops*250/wr_sum/1000;
	rd_speed = 1.0*burst*ops*250/rd_sum/1000;
	cout<<"busrt:"<<burst<<" ops:"<<ops<<" mode:"<<start<<endl;
	
	
	if(start==2){//read
		cout << " dma_read_cmd_counter0: " <<controller -> readReg(525) <<endl;
		cout <<  std::dec << "rd_speed: " << rd_speed << " GB/s" << endl;
		outfile<<rd_speed<<endl;
	}
	if(start==1){//write
		cout << " dma_write_cmd_counter1: " <<controller ->readReg(522) <<endl;
		cout <<  std::dec << "wr_speed: " << wr_speed << " GB/s" << endl;
		outfile<<wr_speed<<endl;
	}
	if(start==3){
		cout << " dma_read_cmd_counter0: " <<controller -> readReg(525) <<endl;
		cout << " dma_write_cmd_counter1: " <<controller ->readReg(522) <<endl;
		cout <<  std::dec << "wr_speed: " << wr_speed << " GB/s" << endl;
		cout <<  std::dec << "rd_speed: " << rd_speed << " GB/s" << endl;
		outfile<<wr_speed<<" "<<rd_speed<<endl;
	}
	outfile.close();
}
void pressure_test(param_test_t param_in,int burst,int ops,int start){
	// cudaDeviceProp device_prop;
	// cudaGetDeviceProperties(&device_prop, 0);
	// printf("GPU最大时钟频率: %.0f MHz (%0.2f GHz)\n",device_prop.clockRate*1e-3f, device_prop.clockRate*1e-6f);
	outfile.open("data.txt", ios::out |ios::app );
	int blocks=2048;
	int threads=1024;
	int total=blocks*threads;
	unsigned int * out;
	unsigned int * out_cpu = new unsigned int[total];
	cudaMalloc(&out,sizeof(unsigned int)*total);

	double elapsedTime;
	struct timeval t_start, t_end;
    gettimeofday(&t_start,NULL);

	int iter = 50000;
	size_t buffer_size = 200*1024*1024;
	gpu_pressure<<<blocks,threads>>>((unsigned int *)param_in.map_d_ptr,iter,buffer_size,out);
	// for(int i=0;i<50000;i++){
	// 	GlobalCopy<<<blocks,threads>>>((int *)out,(int *)param_in.map_d_ptr,200*1024*1024);
	// }
	
	gpu_benchmark(param_in,burst,ops,start);
	cudaThreadSynchronize();
	cudaError_t cudaerr = cudaPeekAtLastError();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
	gettimeofday(&t_end,NULL);
	elapsedTime =  t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
	cout<<"Time:"<<elapsedTime<<endl;
	cout<<"speed"<<1.0*iter*buffer_size/elapsedTime/1024/1024/1024 << endl;
	
	// cudaMemcpy(out_cpu,out,sizeof(unsigned int)*total,cudaMemcpyDeviceToHost);
	// for(int i=0;i<16;i++){
	// 	cout<<out_cpu[i]<<" ";
	// }
	// for(int i=0;i<total;i++){
	// 	if(out_cpu[i]!=out_cpu[0]){
	// 		cout<<"error at:"<<i<<" value:"<<out_cpu[i]<<endl;
	// 		break;
	// 	}
	// }
	printf("###########gpu move done!\n");
	cout<<endl<<endl;
}

void socket_sample(param_interface_socket_t param_in){
	socket_context_t* context = get_socket_context(param_in.buffer_addr,param_in.tlb_start_addr,param_in.controller);

	int * data;
	size_t total_data_length = 4*256*1024*1024;
	cudaMalloc(&data,total_data_length);

	sock_addr_t addr;
	addr.ip = param_in.ip;
	addr.port = param_in.port;

	int* socket1;
	cudaMalloc(&socket1,sizeof(int));

	connection_t* connection1;
	cudaMalloc(&connection1,sizeof(connection_t));

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	sleep(3);
	printf("---start user code:\n");

	int verify_data_offset = 5;
	int transfer_data_length = 40*1024*1024;
	if(app_type==0){
		//client code
		create_socket<<<1,1,0,stream>>>(context,socket1);
		compute<<<1,1024,0,stream>>>(data,total_data_length,verify_data_offset);
		connect<<<1,1,0,stream>>>(context,socket1,addr);
		socket_send<<<1,8,0,stream>>>(context,socket1,data,transfer_data_length);
	}else if(app_type==1){
		//server code
		create_socket<<<1,1,0,stream>>>(context,socket1);
		socket_listen<<<1,1,0,stream>>>(context,socket1,1235);
		accept<<<1,1,0,stream>>>(context,socket1,connection1);
		socket_recv<<<1,8,0,stream>>>(context,connection1,data,transfer_data_length);
		verify<<<1,1,0,stream>>>(data,transfer_data_length,verify_data_offset);
		//socket_close<<<1,1,0,stream>>>(context,connection1);
	}else{
		printf("app_type not set!\n");
	}

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


