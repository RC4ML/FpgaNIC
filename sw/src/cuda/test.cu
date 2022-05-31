#include "test.cuh"
#include "network.cuh"
#include "util.cuh"
#include <time.h>
#include <fstream>
#include <iostream>
#include "sys/time.h"
#include "tool/log.hpp"


using namespace std;

static int mapped=0;
static unsigned int * ctrl_addr0;
static unsigned int * ctrl_addr1;
static unsigned int * bypass_addr0;
static unsigned int * bypass_addr1;

ofstream outfile;
__global__ void GlobalCopy(int *out, const int *in, size_t N )
{
    int temp[16];
	N=size_t(N/4);
	//avoid accessing cache, assure cold-cache access
	int start = (blockIdx.x * blockDim.x + threadIdx.x);
    int step = (blockDim.x * gridDim.x);
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
		cjdebug("###########gpu move start! total_threads:%d  opnum:%ld  iter_num:%ld\n",total_threads,op_num,iter_num);
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
		cjdebug("e-s:%ld  time=%f speed=%f GB/s \n",e-s,time,speed);
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

	if(start == 2){
		cout<<"########### FPGA start reading memory\n";
	}else if(start == 1){
		cout<<"########### FPGA start writing memory\n";
	}
	
	int rd_sum,wr_sum;
	float rd_speed,wr_speed;
	int total_length = 64*1024*1024 ;

	controller ->writeReg(36,total_length);
	controller ->writeReg(37,ops);
	controller ->writeReg(38,burst);
	controller ->writeReg(39,0);
	controller ->writeReg(39,start);
	sleep(10);
  	controller ->writeReg(39,0);
	
	
	rd_sum = controller ->readReg(592);
	wr_sum = controller ->readReg(576);
	// cout << "wr_sum: " << wr_sum <<endl; 
  	// cout << "rd_sum: " << rd_sum <<endl;
	wr_speed = 1.0*burst*ops*250/wr_sum/1000;
	rd_speed = 1.0*burst*ops*250/rd_sum/1000;
	cout<<"busrt:"<<burst<<" Bytes, ops:"<<ops<<" mode:"<<start<<endl;
	
	
	if(start==2){//read
		// cout << "ignore it, dma_read_cmd_counter0: " <<controller -> readReg(525) <<endl;
		cout <<  std::dec << "FPGA read memory speed : " << rd_speed << " GB/s" << endl;
		outfile<<rd_speed<<endl;
	}
	if(start==1){//write
		// cout << "ignore it, dma_write_cmd_counter1: " <<controller ->readReg(522) <<endl;
		// cout<<burst<<" "<<ops<<" "<<wr_sum<<endl;
		cout <<  std::dec << "FPGA write memory speed : " << wr_speed << " GB/s" << endl;
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
__global__ void cal(){
	while(1){

	}
}
void pressure_test(param_test_t param_in,int burst,int ops,int start){
	outfile.open("data.txt", ios::out |ios::app );
	int blocks=1;
	int threads=512;
	int total=blocks*threads;
	unsigned int * out;
	unsigned int * out_cpu = new unsigned int[total];
	cudaMalloc(&out,sizeof(unsigned int)*total);

	// double elapsedTime;
	struct timeval t_start, t_end;
    gettimeofday(&t_start,NULL);

	// int iter = 50000;
	// size_t buffer_size = 200*1024*1024;
	//gpu_pressure<<<blocks,threads>>>((unsigned int *)param_in.map_d_ptr,iter,buffer_size,out);
	cal<<<1,1>>>();
	// for(int i=0;i<50000;i++){
	// 	GlobalCopy<<<blocks,threads>>>((int *)out,(int *)param_in.map_d_ptr,200*1024*1024);
	// }
	
	gpu_benchmark(param_in,burst,ops,start);
	//cudaThreadSynchronize();
	cudaError_t cudaerr = cudaPeekAtLastError();
	ErrCheck(cudaerr);
	gettimeofday(&t_end,NULL);
	// elapsedTime =  t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
	// cout<<"Time:"<<elapsedTime<<endl;
	// cout<<"speed"<<1.0*iter*buffer_size/elapsedTime/1024/1024/1024 << endl;
	
	cjdebug("########### end of this batch!\n");
	cout<<endl<<endl;
}

void test_latency_fpga_cpu(param_test_t param_in){
	ofstream outctrl,outbypass,outdma;
	outctrl.open("latency_ctrl.txt", ios::out |ios::app );
	outbypass.open("latency_bypass.txt", ios::out |ios::app );
	outdma.open("latency_dma.txt", ios::out |ios::app );

	fpga::XDMAController* controller = param_in.controller;
	uint64_t addr = (uint64_t)param_in.tlb_start_addr;
	// cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 
	controller ->writeReg(0,0);
	controller ->writeReg(0,1);//reset
	sleep(1);
	// cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 
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
	// cout << " ctrl lat: " << controller ->readReg(516)  << " cycle" << std::endl;  
	cout << "CPU read FPGA latency: " << 1.0 * controller ->readReg(517) * 4 /1000 << " us" << std::endl; 
	// cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 

	outctrl.close();
	outbypass.close();
	outdma.close();
}

__global__ void test_latency_fpga_gpu_cuda(unsigned int * ctrl_addr0,unsigned int * ctrl_addr1,unsigned int * bypass_addr0,unsigned int * bypass_addr1){
	int index = blockIdx.x*blockDim.x+threadIdx.x;	
	__shared__ unsigned int res0[16];
	__shared__ unsigned int res1[16];
	// unsigned int res_ctrl0;
	// unsigned int res_ctrl1;
	
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
		// printf("bypass latency:%lu\n",e-s);
		printf("ignore it:%d\n",res0[index]);
		printf("ignore it:%d\n",res1[index]);
		(*ctrl_addr0)=(unsigned int)(e-s);
	END_SINGLE_THREAD_DO
	__syncthreads();
	
}

__global__ void read_2080(unsigned int * ctrl_reg,unsigned int * by_reg,size_t length,int *out){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int op_num = int(length/(sizeof(int)));
	int iter_num = int(op_num/total_threads);
	for(int i=0;i<iter_num;i++){
		out[total_threads*i+index] = by_reg[total_threads*i+index];
		//by_reg[total_threads*i+index] = out[total_threads*i+index];
	}
	// by_reg[index] = index;
	// for(int i=0;i<100;i++){
	// 	BEGIN_SINGLE_THREAD_DO
	// 		cu_sleep(5);
	// 		printf("ctrl:%d\n",ctrl_reg[0]);
	// 		ctrl_reg[0] = ctrl_reg[0]+1;
	// 	END_SINGLE_THREAD_DO
	// 	by_reg[index] = by_reg[index]+1;
	// }
}
void test_2080(param_test_t param_in){

	size_t length = 128*1024*1024;
	int *out;
	cudaMalloc(&out,length);
	

	fpga::XDMAController* controller = param_in.controller;
	controller->writeReg(40,length/64);
	// uint64_t* data=(uint64_t*)malloc(64);
	// uint64_t* res=(uint64_t*)malloc(64);
	// for(int i=0;i<8;i++){
	// 	data[i]=i;
	// 	res[i]=0;
	// }
	// controller->writeBypassReg(0,data);
	// controller->readBypassReg(0,res);
	// for(int i=0;i<8;i++){
	// 	printf("%ld ",res[i]);
	// }
	// printf("\n");
	unsigned int * ctrl_reg;
	unsigned int * by_reg;
	ctrl_reg = map_reg_4(500,controller);
	by_reg = map_reg_cj(0,controller,length);
	//read_2080<<<8,1024>>>(ctrl_reg,by_reg,length,out);
	cudaMemcpy(out,by_reg,length,cudaMemcpyDeviceToDevice);
	cudaError_t cudaerr = cudaDeviceSynchronize();
	ErrCheck(cudaerr);
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
	cout << "A100 read FPGA latency : " << 1.0 * controller ->readReg(500) /1.41 / 1000<< " us" << std::endl; //gpu rd fpga, clock of gpu, divide 1.41Ghz for A100
	// cout << " by lat: " << controller ->readReg(517) << " cycle" << std::endl; 
	// cout << " dma lat: " << controller ->readReg(593) << " cycle" << std::endl; 

	// outctrl.close();
	outbypass.close();
	outdma.close();
}


__global__ void cal(char data,int times){
	size_t i = 0;
	int t=0;
	while(1){
		i++;
		if(i%30000000==0){
			printf("caling %c\n",data);
			t++;
			if(t>=times){
				break;
			}
		}
	}
}


__global__ void compute_cuda(volatile int * data){
	size_t s,e;
	int index=0;
	int next_index;
	int res[200];
	long int lat[200];

	for(int i=0;i<200;i++){
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
		for(int i=0;i<200;i++){
			sum+=res[i];
		}
		printf("ignore it, res sum:%d\n",sum);
		for(int i=0;i<200;i++){
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
		// size_t N = size_t(2)*1024*1024;//workspace size
		// size_t size = N*sizeof(int);
		
		// int *data;//pinned
		// cudaMallocHost(&data,size);
		
		// //int *data = (int *)malloc(size);
		
		// int *data_dst;
		// cudaMalloc(&data_dst, size);
		
		// for(int i=0;i<N;i++){
		// 	data[i] = i;
		// }
		// int op_times = int(size/stride);
		// int burst = int(stride/4);
		// clock_gettime(CLOCK_MONOTONIC, &beg);
		// for(int i=0;i<op_times;i++){
		// 	cudaMemcpy(data_dst+i*burst,data+i*burst,stride,cudaMemcpyHostToDevice);
		// }
		
		// cudaDeviceSynchronize();
		// clock_gettime(CLOCK_MONOTONIC, &end);
		// double t = 1.0*(end.tv_nsec-beg.tv_nsec)/(1e9)+(end.tv_sec-beg.tv_sec);//seconds
		// double bw = 1.0*N*4/t/1024/1024/1024;
		// //printf("t:%f\n",t);
		// printf("%f\n",bw);
		// out<<bw<<endl;
		// out.close();
		// cudaFree((void*)data);
		// cudaFree((void*)data_dst);
	}
	{//latency
		ofstream out;
		out.open("latency.txt", ios::out |ios::app );
		size_t N = size_t(4000)*1024*1024;
		volatile int *data;
		cudaMallocHost(&data, N*sizeof(int));

		int next_index=stride/sizeof(int);

		for(int index=0;index<N; ){
			data[index] = next_index;
			index = next_index;
			next_index+=stride/sizeof(int);
		}

		int tmp=0;
		compute_cuda<<<1,1>>>(data);
		cudaDeviceSynchronize();
		// cout<<"A100 read CPU latency: "<<1.0*data[0] / 1.4 /1000<<" us"<<endl;
		// out<<"stride:"<<stride<<endl;
		for(int i=0;i<200;i++){
			cout<<"A100 read CPU latency: "<<1.0*data[i] / 1.41 /1000<<" us"<<endl;
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
	// fpga::XDMAController* controller = param_in.controller;

	cudaStream_t stream1,stream2;
	cudaEvent_t event1,event2;
	cudaStreamCreate(&stream1); 	
	cudaStreamCreate(&stream2);
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	cal<<<1,1,0,stream1>>>('A',5);
	cudaEventRecord(event1, stream1);
	cudaStreamWaitEvent(stream2, event1,0);
	cal<<<1,1,0,stream1>>>('B',5);
	cal<<<1,1,0,stream2>>>('C',5);
	cudaEventRecord(event2, stream2);
	cudaStreamWaitEvent(stream1, event2,0);
	cal<<<1,1,0,stream2>>>('D',5);



}

__global__ void gpu_tp_test(int * src,int *dst,size_t length,double fre,double* speed){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int op_num = int(length/sizeof(int));
	int iter_num = int(op_num/total_threads);
	clock_t s,e;
	s=clock64();
	for(int i=0;i<iter_num;i++){
		dst[total_threads*i+index]	=	src[total_threads*i+index];
	}
	e=clock64();
	if(threadIdx.x==0){
		clock_t cycles=e-s;
		double t = cycles/fre/1e9;
		(*speed) = length/t/1024/1024/1024;
		printf("cycles:%ld t:%f speed:%f\n",cycles,t,*speed);
	}	

}
void test_gpu_throughput(param_test_t param_in){
	int * gpu_direct_mem = (int *)param_in.map_d_ptr;
	int * data;
	size_t total_data_length = 8*1024;
	cudaMalloc(&data,total_data_length);

	double fre = get_fre();

	double *speed;
	cudaMallocManaged(&speed,sizeof(double));

	struct timespec beg, end;
	clock_gettime(CLOCK_MONOTONIC, &beg);
	gpu_tp_test<<<1,32>>>(data,gpu_direct_mem,total_data_length,fre,speed);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);
	double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
	double Bps = total_data_length / dt_ms * 1e3;
	cout << "BW: " << Bps / 1024.0 / 1024.0 << "MB/s" << endl;
	cout<<*speed<<endl;
}