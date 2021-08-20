#include "kernel.cuh"
#include "util.cuh"
#include "tool/log.hpp"
 __global__ void verify(int * data,size_t length,int offset){
	 int count=0;
	 int flag=1;
	 int wrong_index;
	 int wrong_value;
	 int right_value;
	 int count_2m=0;
	 BEGIN_SINGLE_THREAD_DO
		size_t op_num = size_t(length/sizeof(int));
		printf("start verify %ld\n",op_num);
		for(int i=0;i<op_num;i++){
			if(data[i]!=i+offset){
				if(flag==1){
					wrong_index=i;
					wrong_value=data[i];
					right_value=i+offset;
					flag=0;
				}
				count+=1;
				if(i+offset-data[i]==524288){
					count_2m+=1;
				}
			}
		}
		if(flag==1){
			printf("verify data success!\n");
		}else{
			printf("verify data failed!\n");
			printf("index:%d data: %d which should be %d\n",wrong_index,wrong_value,right_value);			
			printf("wrong num: %d  wrong2m: %d\n",count,count_2m);
		}
		
	 END_SINGLE_THREAD_DO
 }

__global__ void compute(int * data,size_t length,int offset){
	int index = blockIdx.x*blockDim.x+threadIdx.x;	
	int total_threads = gridDim.x*blockDim.x;
	size_t op_num = size_t(length/sizeof(int));
	int iter_num = int(op_num/total_threads);

	for(int i=0;i<iter_num;i++){
		data[i*total_threads+index]	=	i*total_threads+index+offset;
	}
	
	BEGIN_SINGLE_THREAD_DO
		cjdebug("function compute done!\n");
	END_SINGLE_THREAD_DO
}

__global__ void movThread(param_cuda_thread_t param){
	int rCount;
	int moveCount=0;
	unsigned int* dataAddr = param.devVAddr0 + int(2*1024*1024/4);
	int step = int(2*1024*1024/4);
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int pagesToRead = int(param.data_length/2/1024/1024);

	rCount = param.devReadCountAddr0[0];
	if(0 == index){
		printf("index:%d rcount:%d devReadCountAddr0: %d   devWriteCountAddr: %d\n",index,rCount,param.devReadCountAddr0[0],param.devWriteCountAddr0[0]);
		printf("index:%d blocks:%d threadsPerBlock:%d pages to read:%d\n",index,param.blocks,param.threadsPerBlock,pagesToRead);
	}
	int tmp = param.devReadCountAddr0[0];
	
	{//ptx test
		
		if(index==1){
			uint64_t volatile addr = (uint64_t)(dataAddr+16*5+16*10*index);
			int volatile y=0;
			uint64_t s=0;
			uint64_t e=0;
			asm volatile(
				"mov.u64 %0,%%clock64;\n\t"
				"ld.u32 %2,[%3];\n\t"//没有就是7 有的话16   ca=16  cg=16   cs=16 lu=16  cv=16
				"mov.u64 %1,%%clock64;\n\t"
				:"=l"(s),"=l"(e),"=r"(y):"l"(addr): "memory"
			);
			printf("s:%ld e:%ld e-s:%ld y:%d\n",s,e,e-s,y);
		}
	}
	__syncthreads();
	while(1){
		// if(index==0){
		// 	param.devReadCountAddr0[0] =param.devWriteCountAddr0[0];
		// 	if(param.devReadCountAddr0[0]==(tmp+1000)){
		// 		break;
		// 	}
		// }
		
		while((index%(blockDim.x))==0 && rCount==param.devWriteCountAddr0[0]){
			continue;
		}
		__syncthreads();
		unsigned int* startDstAddr = param.dstAddr0+step*(rCount%(param.buffer_pages));
		unsigned int* startSrcAddr = dataAddr + step*(rCount%(param.buffer_pages));
		// if(index<10)
		// printf("i:%d %d %d %d\n",index,rCount,step*(rCount%100),moveCount);
		int length = int(step/param.threadsPerBlock/param.blocks);
		int stride = param.threadsPerBlock*param.blocks;
		for(int i=0;i<length;i++){
			startDstAddr[i*stride+index] = startSrcAddr[i*stride+index];
		}
		rCount+=1;
		__syncthreads();
		if(index==0){
			param.devReadCountAddr0[0] = rCount;
			//printf("index:%d  devReadCountAddr:%d   devWriteCountAddr:%d\n",index,param.devReadCountAddr0[0],param.devWriteCountAddr0[0]);
		}
		moveCount+=1;
		if(moveCount==pagesToRead){
			break;
		}
	}
	__syncthreads();
	{//ptx test
		
		if(index==1){
			uint64_t addr = (uint64_t)(dataAddr+16*5+16*3000*index);
			int y=0;
			uint64_t s=0;
			uint64_t e=0;
			asm volatile(
				"mov.u64 %0,%%clock64;\n\t"
				"ld.cv.u32 %2,[%3];\n\t"//没有就是7 有的话16   ca=16  cg=16   cs=16 lu=16  cv=16
				"mov.u64 %1,%%clock64;\n\t"
				:"=l"(s),"=l"(e),"=r"(y):"l"(addr): "memory"
			);
			printf("s:%ld e:%ld e-s:%ld y:%d\n",s,e,e-s,y);

		}
	}
	__syncthreads();
	{//check results
		if(index==0){
			printf("read done! %d %d \n",tmp,param.devReadCountAddr0[0]);
			for(int i=0;i<int(param.data_length/64);i++){
				if(param.dstAddr0[i*16] != i%32768+param.offset){
					printf("%dth with value:%d\n",i,param.dstAddr0[i*16]);
				}
			}
			printf("check done! %d %d \n",tmp,param.devReadCountAddr0[0]);
		}
	}
	

	// {//print data
	// 	if(index==0){
	// 		for(int i=0;i<int(param.data_length/64);i+=1024){
	// 			printf("i:%d  data:%d\n",i,param.dstAddr0[i*16]);
	// 		}
	// 	}
	// }

}




// __global__ void writeBypassReg(volatile unsigned int *dev_addr,int *blocks){
// 	//printf("enter writeBypassReg thread with mapped dev_addr:%x\n",dev_addr);
// 	int index = blockIdx.x*blockDim.x+threadIdx.x;
// 	int stride = blocks[0]*blockDim.x;
// 	int num = int(1024*1024/4/stride);
// 	int sum=0;
// 	for(int i=0;i<num;i++){
// 		dev_addr[i*stride+index]=i;
// 		sum+=dev_addr[i*stride+index];
// 	}
// 	printf("%d %d %d \n",index,stride,num);
// 	printf("%d \n",sum);
// }
// __global__ void readBypassReg(volatile unsigned int *dev_addr,int *blocks){
// 	//printf("enter readBypassReg thread with mapped dev_addr:%x\n",dev_addr);
// 	int index = blockIdx.x*blockDim.x+threadIdx.x;
// 	int next_addr,sum;
	
// 	next_addr=dev_addr[1600+index];
// 	clock_t s = clock64();
// 	sum = dev_addr[next_addr+index];
// 	// __threadfence();
// 	clock_t e = clock64();
// 	printf("latency:%lu\n",e-s);
// 	printf("%d %d \n",next_addr,sum);
// }
__global__ void writeReg(volatile unsigned int *dev_addr,int *blocks){
	
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