#include"inference.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <c10/core/impl/InlineEvent.h>
#include <c10/core/Event.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include<pthread.h>
#include"inference_util.cuh"
#include"main.h"
#include "kernel.cuh"
#include "network.cuh"
using namespace std;

socket_context_t* context_infer;
param_test_t param_in_;
void * sub_thread(void *arg){
	unsigned int* buffer_addr =  ((unsigned int*)param_in_.map_d_ptr);
	printf("buffer_addr_sub_td:%lx\n",(long)buffer_addr);
	context_infer = get_socket_context(buffer_addr,param_in_.tlb_start_addr,param_in_.controller,app_type);
	while(1){
		sleep(3);
		// printf("subthread looping\n");
	};
}
void inference_sample(param_test_t param_in){
	torch::jit::script::Module module;
	std::vector<torch::jit::IValue> inputs;
	if(app_type==1){
		string model_file = "/home/amax4/cj/sw/create_model/model/traced_resnet_model.pt";
		printf("ready to transfer model to gpu\n");
		module = torch::jit::load(model_file,at::kCUDA);//module.to(at::kCUDA,true);
		printf("transfer model done\n");
	}
	param_in_ = param_in;

	pthread_t thread;
	int ret = pthread_create(&thread,NULL,sub_thread,(void*)0);
	if(ret == -1){
		printf("Create pthread error!\n");
		return;
	}
	sleep(3);

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

	if(app_type==0){//client
		float data[10][150528];
		load_data(data);
		float * data_gpu;
		cudaMalloc(&data_gpu,150528*sizeof(float));
		printf("ready memcpy!\n");
		cudaMemcpy(data_gpu,data[1],150528*sizeof(float),cudaMemcpyHostToDevice);

		float * data_res_gpu;
		cudaMalloc(&data_res_gpu,1000*sizeof(float));

		sleep(5);
		printf("start tcp part!\n");
		create_socket<<<1,1,0,stream1>>>(context_infer,socket1);
		connect<<<1,1,0,stream1>>>(context_infer,socket1,addr);
		socket_send<<<1,1024,0,stream1>>>(context_infer,socket1,(int *)data_gpu,150528*sizeof(float));
		cudaEventRecord(event1, stream1);
		cudaStreamWaitEvent(stream2, event1,0);
		socket_recv<<<1,1024,0,stream1>>>(context_infer,socket1,(int *)data_res_gpu,sizeof(float)*1024);
		socket_recv_ctrl<<<1,16,0,stream2>>>(context_infer,socket1,(int *)data_res_gpu,sizeof(float)*1024);
		get_max<<<1,1,0,stream1>>>(data_res_gpu);
	}else if(app_type==1){//server
		printf("start server code\n");
		auto options = torch::TensorOptions().device(torch::kCUDA);
		
		
		

		size_t transfer_length=150528*sizeof(float);
		int * data_gpu;
		cudaMalloc(&data_gpu,transfer_length);
		//recv
		printf("start tcp part!\n");
		create_socket<<<1,1,0,stream1>>>(context_infer,socket1);
		socket_listen<<<1,1,0,stream1>>>(context_infer,socket1,6666);
		accept<<<1,1,0,stream1>>>(context_infer,socket1,connection1);
		cudaEventRecord(event1, stream1);
		cudaStreamWaitEvent(stream2, event1,0);
		socket_recv<<<1,1024,0,stream1>>>(context_infer,connection1,(int *)data_gpu,transfer_length);
		socket_recv_ctrl<<<1,16,0,stream2>>>(context_infer,connection1,(int *)data_gpu,transfer_length);
		cudaStreamSynchronize(stream1);
		// auto s1 = at::cuda::getStreamFromPool();
		// auto s2 = at::cuda::getStreamFromPool();
		// auto s3 = at::cuda::getStreamFromPool();
		// at::cuda::setCurrentCUDAStream(s2);
		// auto defaultStream = at::cuda::getDefaultCUDAStream();
		// cout<<s1<<endl;
		// cout<<s2<<endl;
		// cout<<s3<<endl;
		// cout<<defaultStream<<endl;
		// printf("start inference\n");
		// inputs.push_back(torch::from_blob(data_gpu,{1,3,224,224},options));
		// at::Tensor output = module.forward(inputs).toTensor();
		// float * d = output.data_ptr<float>();

		socket_send<<<1,1024,0,stream1>>>(context_infer,connection1,(int *)data_gpu,sizeof(float)*1024);

		// tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(output, 1);
		// int max_index_int = std::get<1>(max_classes)[0].item<int>();
		// cout << max_index_int << endl;

		// ifstream labels_file("/home/amax4/cj/sw/create_model/classes.txt");
		// vector<string> labels;
		// string line;
		// while (std::getline(labels_file, line)){
		// 	labels.push_back(string(line));
		// }
		// cout<<labels[max_index_int]<<endl;
	}
}