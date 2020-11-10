#include "test.cuh"
#include "network.cuh"
#include "util.cuh"
#include <time.h>
#include <fstream>
#include <iostream>


using namespace std;

static int mapped=0;
static unsigned int * ctrl_addr0;
static unsigned int * ctrl_addr1;
static unsigned int * bypass_addr0;
static unsigned int * bypass_addr1;
void test_latency_fpga_cpu(param_test_t param_in){
	ofstream outctrl,outbypass,outdma;
	outctrl.open("latency_ctrl.txt", ios::out |ios::app );
	outbypass.open("latency_bypass.txt", ios::out |ios::app );
	outdma.open("latency_dma.txt", ios::out |ios::app );

	fpga::XDMAController* controller = param_in.controller;
	uint64_t addr = (uint64_t)param_in.tlb_start_addr;
	cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 
	controller ->writeReg(0,0);
	controller ->writeReg(0,1);//reset
	sleep(1);
	cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 
	int ctrl_addr;
	ctrl_addr = controller ->readReg(500);
	controller ->readReg(ctrl_addr);

	uint64_t by_addr[8];
	uint64_t res[8];
	controller ->readBypassReg(31,by_addr);
	controller ->readBypassReg(by_addr[0],res); 

	controller ->writeReg(32,(uint32_t)addr);     //rd base addr
	controller ->writeReg(33,(uint32_t)(addr>>32));
	controller ->writeReg(36,10*1024*1024);//dma buffer length
	controller ->writeReg(37,1);
	controller ->writeReg(38,2*1024*1024);
	controller ->writeReg(39,0);
	controller ->writeReg(39,2);

	sleep(1);
	outctrl<<controller ->readReg(516)<<endl;
	outbypass<<controller ->readReg(517)<<endl;
	outdma<<controller ->readReg(593)<<endl;
	cout << " ctrl lat: " << controller ->readReg(516)  << " cycle" << std::endl;  
	cout << " by lat: " << controller ->readReg(517) << " cycle" << std::endl; 
	cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 

	outctrl.close();
	outbypass.close();
	outdma.close();
}

__global__ void test_latency_fpga_gpu_cuda(unsigned int * ctrl_addr0,unsigned int * ctrl_addr1,unsigned int * bypass_addr0,unsigned int * bypass_addr1){
	int index = blockIdx.x*blockDim.x+threadIdx.x;	
	__shared__ unsigned int res0[16];
	__shared__ unsigned int res1[16];
	unsigned int res_ctrl0;
	unsigned int res_ctrl1;
	
	// BEGIN_SINGLE_THREAD_DO// read ctrl reg
	// 	res_ctrl0 = (*ctrl_addr0);
	// 	if(res_ctrl0==0){
	// 		res_ctrl1 = (*ctrl_addr1);
	// 	}
	// 	printf("res_ctrl0:%d\n",res_ctrl0);
	// 	printf("res_ctrl1:%d\n",res_ctrl1);
	// END_SINGLE_THREAD_DO

	
	// BEGIN_SINGLE_THREAD_DO//write ctrl reg
	// 	(*ctrl_addr0) = 3;
	// 	(*ctrl_addr1) = 5;
	// END_SINGLE_THREAD_DO


	res0[index] = bypass_addr0[index];
	clock_t s = clock64();
	if(res0[index]==0){
		res1[index] = bypass_addr1[index];
	}
	clock_t e = clock64();
	
	BEGIN_SINGLE_THREAD_DO
		printf("bypass latency:%lu\n",e-s);
		printf("bypass_ctrl0:%d\n",res0[index]);
		printf("bypass_ctrl1:%d\n",res1[index]);
		(*ctrl_addr0)=(unsigned int)(e-s);
	END_SINGLE_THREAD_DO
	__syncthreads();
	
}
void test_latency_fpga_gpu(param_test_t param_in){
	ofstream outctrl,outbypass,outdma;
	// outctrl.open("latency_ctrl.txt", ios::out |ios::app );
	outbypass.open("latency_bypass.txt", ios::out |ios::app );
	outdma.open("latency_dma.txt", ios::out |ios::app );

	fpga::XDMAController* controller = param_in.controller;

	if(mapped==0){
		ctrl_addr0 = map_reg_4(500,controller);
		ctrl_addr1 = map_reg_4(504,controller);
		bypass_addr0 = map_reg_64(31,controller);
		bypass_addr1 = map_reg_64(0,controller);
		mapped=1;
	}

	uint64_t addr = (uint64_t)param_in.tlb_start_addr;
	controller ->writeReg(0,0);
	controller ->writeReg(0,1);//reset
	sleep(1);

	test_latency_fpga_gpu_cuda<<<1,16>>>(ctrl_addr0,ctrl_addr1,bypass_addr0,bypass_addr1);
	controller ->writeReg(32,(uint32_t)addr);     //rd base addr
	controller ->writeReg(33,(uint32_t)(addr>>32));
	controller ->writeReg(36,10*1024*1024);//dma buffer length
	controller ->writeReg(37,1);
	controller ->writeReg(38,2*1024*1024);
	controller ->writeReg(39,0);
	controller ->writeReg(39,2);

	sleep(1);
	//outctrl<<controller ->readReg(516)<<endl;
	outbypass<<controller ->readReg(500)<<endl;
	outdma<<controller ->readReg(593)<<endl;
	// cout << " ctrl lat: " << controller ->readReg(516)  << " cycle" << std::endl; 
	cout << " by lat: " << controller ->readReg(500) << " cycle" << std::endl; 
	cout << " by lat: " << controller ->readReg(517) << " cycle" << std::endl; 
	cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 

	// outctrl.close();
	outbypass.close();
	outdma.close();
}


__global__ void init_mem(int * addr,int *addr_cpu,int offset){
	for(int i=0;i<200*1024*1024/4;i++){
		if(addr[i]!=i+offset){
			printf("gpu mem %d %d\n",i,addr[i]);
			break;
		}
		// if(addr_cpu[i]!=i+offset){
		// 	printf("cpu mem %d %d\n",i,addr_cpu[i]);
		// 	break;
		// }
	}
	printf("init mem done!\n");
}
__global__ void cal(){
	size_t i = 0;
	printf("caling\n");
	while(1){
		i++;
		if(i%100000000==0){
			printf("looping\n");
		}
	}
}
void test_cpu_gpu(param_test_t param_in){
	//cal<<<1,1>>>();
	int num = 50*1024*1024;
	int size = 200*1024*1024;
	int *cpu_mem=(int *)malloc(200*1024*1024);
	int *gpu_mem=(int *)param_in.map_d_ptr;
	int *gpu_mem_cpu_ptr=(int *)param_in.d_mem_cpu;
	int * cpu_mem_gpu_ptr;

	cudaError_t err = cudaHostRegister(cpu_mem,200*1024*1024,cudaHostRegisterMapped);
	ErrCheck(err);
	cudaHostGetDevicePointer((void **) &(cpu_mem_gpu_ptr), cpu_mem, 0);
	int offset=5;
	for(int i=0;i<200*1024*1024/4;i++){
		gpu_mem_cpu_ptr[i]=offset+i;
		cpu_mem[i]=offset+i;
	}
	//init_mem<<<1,1>>>(gpu_mem,cpu_mem_gpu_ptr,offset);

	printf("start test cpu and gpu!\n");
	struct timespec beg, end;

	int *cpu_buf=(int *)malloc(2000*1024*1024);

	printf("start copy\n");

	{
	clock_gettime(CLOCK_MONOTONIC, &beg);
		memcpy(gpu_mem_cpu_ptr,cpu_mem,size);
		//memcpy(cpu_mem,gpu_mem_cpu_ptr,size);
	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("end copy\n");
	}

	double byte_count = (double) size;
	double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
	double Bps = byte_count / dt_ms * 1e3;
	cout << "BW: " << Bps / 1024.0 / 1024.0 << "MB/s" << endl;

}

__global__ void compute_cuda(volatile int * data){
	size_t s,e;
	int index=0;
	int next_index;
	int res[100];
	long int lat[100];

	for(int i=0;i<100;i++){
		s = clock64();
		next_index = data[index];
		res[i] = next_index;
		e = clock64();
		lat[i] = e-s;
		index=next_index;

		//printf("latency:%lu next_index:%d\n",lat[i],next_index);
		
	}
	BEGIN_SINGLE_THREAD_DO
		int sum=0;
		for(int i=0;i<100;i++){
			sum+=res[i];
		}
		printf("res sum:%d\n",sum);
		for(int i=0;i<100;i++){
			data[i]=lat[i];
		}
	END_SINGLE_THREAD_DO
}

// __global__ void compute_cuda_bw(volatile int * data,int N,long int* res){
// 	int sum=0;
// 	int index = blockIdx.x*blockDim.x+threadIdx.x;	
// 	size_t op_num = N;
// 	int total_threads = blockDim.x*gridDim.x;
// 	size_t iter_num = size_t(op_num/total_threads);

// 	clock_t s,e;
// 	s= clock64();
// 	__syncthreads();
// 	for(int i=0;i<iter_num;i++){
// 		sum+=data[total_threads*i+index];
// 	}
// 	__syncthreads();
// 	e=clock64();
// 	BEGIN_SINGLE_THREAD_DO
// 		res[0]=e-s;
// 		printf("%lu\n",e-s);
// 		printf("total threads:%d\n",total_threads);
// 	END_SINGLE_THREAD_DO
	
// }
void test_simple(int stride){
	{//throughput
		// ofstream out;
		// out.open("bw.txt", ios::out |ios::app );
		// struct timespec beg, end;
		// size_t N = size_t(1000)*1024*1024;
		// size_t size = N*sizeof(int);
		
		// int *data;//pinned
		// cudaMallocHost(&data,size);
		
		// //int *data = (int *)malloc(size);
		
		// int *data_dst;
		// cudaMalloc(&data_dst, size);
		
		// for(int i=0;i<N;i++){
		// 	data[i] = i;
		// }
		// clock_gettime(CLOCK_MONOTONIC, &beg);
		// cudaMemcpy(data_dst,data,size,cudaMemcpyHostToDevice);
		// cudaDeviceSynchronize();
		// clock_gettime(CLOCK_MONOTONIC, &end);
		// double t = 1.0*(end.tv_nsec-beg.tv_nsec)/(1e9)+(end.tv_sec-beg.tv_sec);//seconds
		// double bw = 1.0*N*4/t/1024/1024/1024;
		// printf("t:%f\n",t);
		// printf("bw:%f\n",bw);
		// out<<bw<<endl;
		// out.close();
		// cudaFree((void*)data);
		// cudaFree((void*)data_dst);
	}
	{//latency
		ofstream out;
		out.open("latency.txt", ios::out |ios::app );
		struct timespec beg, end;
		size_t N = size_t(1000)*1024*1024;
		size_t size = N*sizeof(int);
		volatile int *data;
		cudaMallocManaged(&data, N*sizeof(int));

		int next_index=stride/sizeof(int);

		for(int index=0;index<N; ){
			data[index] = next_index;
			index = next_index;
			next_index+=stride/sizeof(int);
		}

		int tmp=0;
		compute_cuda<<<1,1>>>(data);
		cudaDeviceSynchronize();
		out<<"stride:"<<stride<<endl;
		for(int i=0;i<100;i++){
			out<<data[i]<<endl;
		}
		out.close();
	}

}

__global__ void cp(int * data,size_t length,int offset){
	int index = blockIdx.x*blockDim.x+threadIdx.x;	
	int total_threads = gridDim.x*blockDim.x;
	size_t op_num = size_t(length/sizeof(int));
	int iter_num = int(op_num/total_threads);

	for(int i=0;i<iter_num;i++){
		data[i*total_threads+index]	=	i*total_threads+index+offset;
	}
	
	BEGIN_SINGLE_THREAD_DO
		printf("function compute done!\n");
	END_SINGLE_THREAD_DO
}

__global__ void mk(int *dst,int * src,int length){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int op_num = int(length/sizeof(int));
	int total_threads = blockDim.x;
	int iter_num = int(op_num/total_threads);

	for(int i=0;i<iter_num;i++){
		dst[total_threads*i+index]		=	src[total_threads*i+index];
	}
}


void cj_debug(param_test_t param_in){
	fpga::XDMAController* controller = param_in.controller;
	int * data;
	size_t total_data_length = 2*1024*1024;
	cudaMalloc(&data,total_data_length);

	int * gpu_buf = (int *)param_in.map_d_ptr;
	cp<<<1,1024>>>(data,total_data_length,3);
	sleep(5);
	mk<<<1,1024>>>(gpu_buf,data,total_data_length);

}