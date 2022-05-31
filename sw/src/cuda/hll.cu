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

__global__ void hll_init_data(size_t length,volatile unsigned int *data){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	size_t op_num = length/sizeof(int);
	int iter_num = int(op_num/total_threads);
	for(int i=0;i<iter_num;i++){
		data[i*total_threads+index]=i*total_threads+index;
	}
}
__global__ void hll_raw_test(socket_context_t* ctx,size_t length,volatile unsigned int *data){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int block_id = blockIdx.x;

	size_t op_num = length/sizeof(int);
	int iter_num = int(op_num/total_threads);
	size_t start_time,end_time;
	BEGIN_BLOCK_ZERO_DO
		start_time = clock64();
	END_BLOCK_ZERO_DO
	for(int i=0;i<iter_num;i++){
		hll(data+total_threads*i+index,ctx->hll_mem);
	}
	BEGIN_BLOCK_ZERO_DO
		end_time = clock64();
		double speed = 1.0*length/1024/1024/1024/((end_time-start_time)/1.41/1e9);
		printf("speed: %f GB/s\n",speed);
	END_BLOCK_ZERO_DO
	
}

__global__ void init_count_hll_simple(int *mem){
	for(int i=0;i<65536;i++){
		mem[i]=0;
	}
}


__global__ void hll_simple(int* mem, size_t length,volatile unsigned int *data,volatile unsigned int* count){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int block_id = blockIdx.x;

	int batch_size = 8;

	size_t op_num = length/sizeof(int);
	int iter_num = int(op_num/total_threads);
	size_t start_time=0,end_time;

	unsigned int cur_count=0;
	int value = 0+index;

	for(int i=0;i<iter_num/batch_size;i++){
		BEGIN_BLOCK_ZERO_DO
		while(((*count)-cur_count<(total_threads*4))){

		}
		if(start_time ==0){
			start_time = clock64();
		}
		// printf("%d\n",cur_count);
		cur_count+=(total_threads*4*batch_size);
		END_BLOCK_ZERO_DO
		for(int j=0;j<batch_size;j++){
			volatile unsigned int* addr = data+total_threads*(i*batch_size+j)+index;
			// int res = *(addr);
			// if(value!=res){
			// 	printf("error %d %d\n",value,res);
			// }
			// value+=total_threads;
			hll(addr,mem);
		}
		
	}
	BEGIN_BLOCK_ZERO_DO
		end_time = clock64();
		printf("%d\n",cur_count);
		double speed = 1.0*length/1024/1024/1024/((end_time-start_time)/1.41/1e9);
		printf("speed: %f\n",speed);
	END_BLOCK_ZERO_DO
	
}

void hll_simple_dma_benchmark(param_test_t param_in){
	printf("start hll_simple_dma_benchmark\n");
	unsigned int* device_addr_start = ((unsigned int*)param_in.map_d_ptr);
	fpga::XDMAController* controller = param_in.controller;

	unsigned long tlb_start = (unsigned long)device_addr_start;

	size_t length = size_t(1)*1024*1024*1024;

	controller ->writeReg(0,0);
	controller ->writeReg(0,1);//reset
	sleep(1);
	controller ->writeReg(34,(unsigned int)tlb_start);//low
	controller ->writeReg(35,(unsigned int)(tlb_start>>32));
	controller ->writeReg(36,length+2*1024*1024); //total len
	controller ->writeReg(37,1);//ops 1
	controller ->writeReg(40,200);//lat cycles
	
	unsigned int* reg = map_reg_4(577,controller);

	
	int *mem;
	cudaMalloc(&mem,sizeof(int)*65536);
	init_count_hll_simple<<<1,1>>>(mem);

	

	hll_simple<<<256,512>>>(mem,length,device_addr_start,reg);

	sleep(1);
	controller ->writeReg(38,length); //len byte
	controller ->writeReg(39,0);
	controller ->writeReg(39,1); //start
	sleep(1);

	cudaDeviceSynchronize();

	verify<<<1,1>>>((int *)device_addr_start,length,0);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	sleep(2);
	uint t = controller->readReg(576);
	cout<<length<<endl;
	cout<<t<<endl;
	cout<<"speed:"<<1.0*length/1024/1024/1024/(1.0*t*4/1000/1000/1000)<<endl;


}
void hll_sample(param_test_t param_in){

	unsigned int* buffer_addr =  ((unsigned int*)param_in.map_d_ptr);
	socket_context_t* context = get_socket_context(buffer_addr,param_in.tlb_start_addr,param_in.controller,app_type);
	
	sock_addr_t addr;//6 -> 4
	addr.ip = 0xc0a8bd0d;//0a => amax4
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

	{//raw test
		volatile unsigned int *data;
		size_t length = size_t(200)*1024*1024;
		cudaMalloc(&data,length);
		int *mem;
		cudaMalloc(&mem,sizeof(int)*65536);
		init<<<1,1>>>(context,mem);
		hll_init_data<<<1,1024>>>(length,data);
		hll_raw_test<<<hll_sm_num,512>>>(context,length,data);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		return;
	}
	// int verify_data_offset = 5;
	// size_t transfer_length = size_t(50)*1024*1024*4;
	// param_in.controller->writeReg(165,(unsigned int)(transfer_length/64));//count code
	// param_in.controller->writeReg(183,(unsigned int)(transfer_length/64));
	// if(app_type==0){
	// 	sleep(3);
	// 	int * data;
	// 	cudaMalloc(&data,transfer_length);
	// 	compute<<<1,1024,0,stream1>>>(data,transfer_length,verify_data_offset);

	// 	create_socket<<<1,1,0,stream1>>>(context,socket1);
	// 	connect<<<1,1,0,stream1>>>(context,socket1,addr);
	// 	socket_send_pre<<<1,1,0,stream1>>>(context,socket1,transfer_length);
	// 	socket_send<<<2,1024,0,stream1>>>(context,socket1,data,transfer_length);
	// }else if(app_type==1){
	// 	int *mem;
	// 	cudaMalloc(&mem,sizeof(int)*65536);
	// 	init<<<1,1>>>(context,mem);

	// 	int * data_recv;
	// 	cudaMalloc(&data_recv,transfer_length);
	// 	create_socket<<<1,1,0,stream1>>>(context,socket1);
	// 	socket_listen<<<1,1,0,stream1>>>(context,socket1,6666);
	// 	accept<<<1,1,0,stream1>>>(context,socket1,connection1);
	// 	cudaEventRecord(event1, stream1);
	// 	cudaStreamWaitEvent(stream2, event1,0);
	// 	socket_recv<<<4,1024,0,stream1>>>(context,connection1,(int *)data_recv,transfer_length);
	// 	socket_recv_ctrl<<<1,16,0,stream2>>>(context,connection1,(int *)data_recv,transfer_length);
	// 	verify<<<1,1,0,stream1>>>(data_recv,transfer_length,verify_data_offset);
	// }

}
