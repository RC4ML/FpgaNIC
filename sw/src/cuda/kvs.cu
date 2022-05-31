#include"kvs.cuh"
#include "util.cuh"
#include "network.cuh"

#define KEY_NUM_ALIGN(size) (((size)+15) & (~15))

// asm volatile(
// 	"ld.u32.cv %0, [%1];\n\t "
// 	:"=r"(count_value):"l"(count_addr):"memory"
// );

// asm volatile(
// 	"st.u16.wt [%0], %2;\n\t "
// 	"st.u16.wt [%1], %3;\n\t "
// 	:"+l"(send_addr_key),"+l"(send_addr_session):"h"(key_len),"h"(session_id):"memory"
// );

__device__ unsigned int MurMurHash_32(volatile unsigned int *key){
	int len = 4;
	uint32_t seed =  0xbaadf00d;
	uint32_t c1 = 0xcc9e2d51;
	uint32_t c2 = 0x1b873593;
	uint32_t r1 = 15;
	uint32_t r2 = 13;
	uint32_t m = 5;
	uint32_t n = 0xe6546b64;
	uint32_t h = 0;
	uint32_t k = 0;
	uint8_t *d = (uint8_t *) key; // 32 bit extract from `key'
	const uint32_t *chunks = NULL;
	const uint8_t *tail = NULL; // tail - last 8 bytes
	int i = 0;
	int l = len / 4; // chunk length

	h = seed;

	chunks = (const uint32_t *) (d + l * 4); // body
	tail = (const uint8_t *) (d + l * 4); // last 8 byte chunk of `key'

	// for each 4 byte chunk of `key'
	for (i = -l; i != 0; ++i) {
	// next 4 byte chunk of `key'
	k = chunks[i];

	// encode next 4 byte chunk of `key'
	k *= c1;
	k = (k << r1) | (k >> (32 - r1));
	k *= c2;

	// append to hash
	h ^= k;
	h = (h << r2) | (h >> (32 - r2));
	h = h * m + n;
	}

	k = 0;

	// remainder
	switch (len & 3) { // `len % 4'
	case 3: k ^= (tail[2] << 16);
	case 2: k ^= (tail[1] << 8);

	case 1:
		k ^= tail[0];
		k *= c1;
		k = (k << r1) | (k >> (32 - r1));
		k *= c2;
		h ^= k;
	}

	h ^= len;

	h ^= (h >> 16);
	h *= 0x85ebca6b;
	h ^= (h >> 13);
	h *= 0xc2b2ae35;
	h ^= (h >> 16);

	return h;
}

__device__ volatile unsigned int g_mutex;

__global__ void kvs_single_block(
					volatile unsigned int * recv_addr,
					volatile unsigned int * send_addr,
					unsigned int batch_size,
					volatile unsigned int* reg_send_addr_high,
					volatile unsigned int* reg_send_addr_low,
					volatile unsigned int* reg_send_start,
					volatile unsigned int* reg_send_len,
					volatile unsigned int* reg_recv_package_count
){
	// int index = threadIdx.x;
	// int total_threads = blockDim.x*gridDim.x;
	// __shared__ unsigned int recv_package_count;
	// __shared__ volatile unsigned int* recv_head;
	// __shared__ volatile unsigned int* send_head;
	// __shared__ unsigned short key_num_aligned;
	// __shared__ unsigned short infos[64];
	// __shared__ volatile unsigned int* data_addr;
	// __shared__ volatile unsigned int* send_data_addr;


	// unsigned int cmd_count;

	// BEGIN_BLOCK_ZERO_DO
	// 	recv_package_count = *reg_recv_package_count;
	// 	recv_head = recv_addr;
	// 	send_head = send_addr;
	// 	cmd_count = *reg_send_start;
	// END_BLOCK_ZERO_DO

	// while(1){
	// 	BEGIN_BLOCK_ZERO_DO
	// 		while((*reg_recv_package_count) - recv_package_count < batch_size){
	// 		}
	// 		recv_package_count += batch_size;
	// 	END_BLOCK_ZERO_DO
	// 	if(index < 2*batch_size){
	// 		unsigned short * info_addr = (unsigned short*)recv_head;
	// 		infos[index] = info_addr[index];
	// 		volatile unsigned short * send_info_addr = (volatile unsigned short*)(send_head);
	// 		send_info_addr[index] =  index%2==0 ? infos[index] : 4*KEY_NUM_ALIGN(infos[index]);
	// 	}
	// 	for(int i=0;i<batch_size;i++){
	// 		unsigned short key_num = infos[2*i+1];
	// 		key_num_aligned = KEY_NUM_ALIGN(key_num);
	// 		data_addr = recv_head + batch_size + i*key_num_aligned;
	// 		send_data_addr = send_head + batch_size + i*key_num_aligned;
			
	// 		int iter_num = (key_num_aligned-1)/total_threads + 1;
	// 		for(int j=0;j<iter_num;j++){
	// 			unsigned int res = data_addr[j*total_threads+index];
	// 			if(j*total_threads+index < key_num_aligned){
	// 				send_data_addr[j*total_threads+index] = res * 2;
	// 			}
				
	// 		}
			
	// 	}
	// 	BEGIN_BLOCK_ZERO_DO
	// 		recv_head += (batch_size + key_num_aligned*batch_size);

	// 		*reg_send_addr_high = (unsigned int)(((unsigned long)send_head)>>32);
	// 		*reg_send_addr_low = (unsigned int)(send_head);
	// 		unsigned int total_len = batch_size*4 + batch_size*key_num_aligned*4;
	// 		*reg_send_len = total_len;
	// 		cmd_count+=1;
	// 		*reg_send_start = cmd_count;
	// 		send_head += (batch_size + key_num_aligned*batch_size);
	// 	END_BLOCK_ZERO_DO
	// }
}

__global__ void kvs(
					unsigned int * reg_button,
					unsigned int * db_start,
					unsigned int * atomic_count,
					int * timer,
					volatile unsigned int * recv_addr,
					volatile unsigned int * send_addr,
					unsigned int batch_size,
					volatile unsigned int* reg_send_addr_high,
					volatile unsigned int* reg_send_addr_low,
					volatile unsigned int* reg_send_start,
					volatile unsigned int* reg_send_len,
					volatile unsigned int* reg_recv_package_count
					){
	// int index = blockIdx.x*blockDim.x+threadIdx.x;
	// int index_in_block = threadIdx.x;
	// int block_threads = blockDim.x;
	// int total_threads = blockDim.x*gridDim.x;
	// int block_id = blockIdx.x;
	// int block_num = gridDim.x;
	// __shared__ unsigned int recv_package_count;
	// __shared__ volatile unsigned int* recv_head;
	// __shared__ volatile unsigned int* send_head;
	// __shared__ unsigned short key_num;
	// __shared__ unsigned short key_num_aligned;
	// __shared__ unsigned short session_id;
	// __shared__ volatile unsigned int* data_addr;
	// __shared__ volatile unsigned int* send_data_addr;
	// __shared__ unsigned short infos[64];

	// __shared__ unsigned short key_num_last;
	// __shared__ unsigned int send_head_last[32];

	// BEGIN_BLOCK_ZERO_DO
	// 	recv_package_count = *reg_recv_package_count;
	// 	recv_head = recv_addr;
	// 	send_head = send_addr;
	// 	if(block_id==0){
	// 		// *atomic_count = (*reg_send_start)*batch_size;
	// 		atomicAdd((int*) &g_mutex,(*reg_send_start)*batch_size);
	// 	}
	// END_BLOCK_ZERO_DO

	// while(1){
	// 	BEGIN_BLOCK_ZERO_DO
	// 		while((*reg_recv_package_count) - recv_package_count < batch_size){
	// 		}
	// 		recv_package_count += batch_size;
	// 	END_BLOCK_ZERO_DO

	// 	if(block_id==0 && index<batch_size*2){

	// 		unsigned short * info_addr = (unsigned short*)recv_head;
	// 		infos[index] = info_addr[index];
	// 		volatile unsigned short * send_info_addr = (volatile unsigned short*)(send_head);
	// 		send_info_addr[index] =  index%2==0 ? infos[index] : 4*KEY_NUM_ALIGN(infos[index]);
	// 	}

	// 	BEGIN_BLOCK_ZERO_DO
	// 		unsigned short * info_addr = (unsigned short*)(recv_head+block_id);
	// 		if(*reg_button == 1){
	// 			printf("last_key_num:%d \n",key_num_last);
	// 		}
	// 		key_num = info_addr[1];
	// 		key_num_last = key_num;
	// 		key_num_aligned = KEY_NUM_ALIGN(key_num);
	// 		send_data_addr = send_head+batch_size + key_num_aligned*block_id;
	// 		data_addr = recv_head + batch_size + key_num_aligned*block_id;
	// 		recv_head += (batch_size + key_num_aligned*batch_size);
	// 	END_BLOCK_ZERO_DO

	// 	int iter_num = (key_num-1)/block_threads + 1;
	// 	for(int i=0;i<iter_num;i++){
	// 		volatile unsigned int * data_addr_thread = data_addr + i*block_threads + index_in_block;
	// 		volatile unsigned int * write_addr_thread = send_data_addr + i*block_threads + index_in_block;

	// 		// unsigned int res = 2*(*data_addr_thread);
	// 		// *write_addr_thread = res;

	// 		unsigned int res = MurMurHash_32(data_addr_thread);
	// 		unsigned int value = db_start[res%(1024*1024*1024/4)];
	// 		*write_addr_thread = value;
	// 	}
	// 	__threadfence_system();
	// 	__threadfence();

	// 	BEGIN_BLOCK_ZERO_DO	

	// 		while(block_id==0 && ((g_mutex+1)%batch_size)!=0){

	// 		}
	// 		if(block_id==0){
	// 			unsigned int total_len = batch_size*4 + batch_size*key_num_aligned*4;
	// 			*reg_send_len = total_len;
	// 			*reg_send_addr_high = (unsigned int)(((unsigned long)send_head)>>32);
	// 			*reg_send_addr_low = (unsigned int)(send_head);
	// 			*reg_send_start = (g_mutex+1)/batch_size;
	// 			// cu_sleep(2);
	// 		}
	// 		atomicAdd((int*) &g_mutex, 1);
			
	// 		while((g_mutex) % batch_size != 0){

	// 		};
			
	// 		send_head += (batch_size + key_num_aligned*block_num);
	// 	END_BLOCK_ZERO_DO
	// }
	// printf("%d\n",*timer);
}

__global__ void kvs_decouple(
				volatile unsigned int* reg_fifo_cnt802,
				volatile unsigned int* reg_fifo_cnt803,
				unsigned int * db_start,
				volatile unsigned int * recv_addr,
				volatile unsigned int * send_addr,
				unsigned int batch_size,
				unsigned int key_num,
				volatile unsigned int* reg_send_addr_high,
				volatile unsigned int* reg_send_addr_low,
				volatile unsigned int* reg_send_start,
				volatile unsigned int* reg_send_len,
				volatile unsigned int* reg_recv_package_count
				){
	// int index = blockIdx.x*blockDim.x+threadIdx.x;
	// int index_in_block = threadIdx.x;
	// int block_threads = blockDim.x;
	// int total_threads = blockDim.x*gridDim.x;
	// int block_id = blockIdx.x;
	// int block_num = gridDim.x;

	// int iter_num_pre = (batch_size*2-1)/block_threads+1;
	// int packet_per_block = batch_size/block_num;

	// __shared__ unsigned int recv_package_count;
	// __shared__ volatile unsigned int* recv_head;
	// __shared__ volatile unsigned int* send_head;
	// __shared__ unsigned short key_num_aligned;
	// __shared__ unsigned short session_id;
	// __shared__ volatile unsigned int* recv_data_addr;
	// __shared__ volatile unsigned int* send_data_addr;

	// BEGIN_BLOCK_ZERO_DO
	// 	key_num_aligned = KEY_NUM_ALIGN(key_num);
	// 	recv_package_count = *reg_recv_package_count;
	// 	recv_head = recv_addr;
	// 	send_head = send_addr;
	// 	if(block_id==0){
	// 		atomicAdd((int*) &g_mutex,(*reg_send_start)*block_num);
	// 	}
	// END_BLOCK_ZERO_DO
	// while(1){
	// 	BEGIN_BLOCK_ZERO_DO
	// 		while((*reg_recv_package_count) - recv_package_count < batch_size){
	// 		}
	// 		recv_package_count += batch_size;
	// 	END_BLOCK_ZERO_DO

	// 	if(block_id==0){
	// 		unsigned short *	recv_info_addr = (unsigned short*)recv_head;
	// 		volatile unsigned short *	send_info_addr = (volatile unsigned short*)(send_head);
	// 		// if(index==0){
	// 		// 	printf("recv_info_addr:%lx\n",recv_info_addr);
	// 		// 	for(int i=0;i<512;i++){
	// 		// 		printf("%x ",recv_info_addr[i]);
	// 		// 		if(i%16==0){
	// 		// 			printf("\n");
	// 		// 		}
	// 		// 	}
	// 		// 	printf("send_info_addr:%lx\n",send_info_addr);
	// 		// 	for(int i=0;i<512;i++){
	// 		// 		printf("%x ",send_info_addr[i]);
	// 		// 		if(i%16==0){
	// 		// 			printf("\n");
	// 		// 		}
	// 		// 	}
	// 		// }
			

	// 		for(int i=0;i<iter_num_pre;i++){
	// 			int cur_index = i*block_threads+index;
	// 			if(cur_index<batch_size*2){
	// 				send_info_addr[cur_index] = recv_info_addr[cur_index];
	// 			}
	// 		}
	// 	}
	// 	BEGIN_BLOCK_ZERO_DO
	// 		send_data_addr = send_head + batch_size + key_num_aligned*packet_per_block*block_id;
	// 		recv_data_addr = recv_head + batch_size + key_num_aligned*packet_per_block*block_id;
	// 		recv_head += (batch_size + key_num_aligned*batch_size);
	// 	END_BLOCK_ZERO_DO

	// 	int iter_num = (key_num_aligned*packet_per_block-1)/block_threads + 1;
	// 	for(int i=0;i<iter_num;i++){
	// 		int offset = i*block_threads + index_in_block;
	// 		volatile unsigned int * read_addr_thread  = recv_data_addr + offset;
	// 		volatile unsigned int * write_addr_thread = send_data_addr + offset;
	// 		if(offset < key_num_aligned*packet_per_block){
	// 			// unsigned int res = 2*(*read_addr_thread);
	// 			// *write_addr_thread = res;

	// 			unsigned int res = MurMurHash_32(read_addr_thread);
	// 			unsigned int value = db_start[res%(1024*1024*1024/4)];
	// 			*write_addr_thread = value;
	// 		}
			
	// 	}

	// 	__threadfence_system();
	// 	__threadfence();

	// 	BEGIN_BLOCK_ZERO_DO
	// 		while(block_id==0 && ((g_mutex+1)%block_num)!=0){
	// 		}
	// 		while(block_id==0 && (*reg_fifo_cnt802>600 || *reg_fifo_cnt803>256)){
	// 		}
	// 		if(block_id==0){
	// 			// printf("send_head:%lx\n",send_head);
	// 			// for(int i=0;i<512;i++){
	// 			// 	printf("%x ",send_head[i]);
	// 			// 	if(i%16==0){
	// 			// 		printf("\n");
	// 			// 	}
	// 			// }
	// 			unsigned int total_len = batch_size*4 + batch_size*key_num_aligned*4;
	// 			*reg_send_len = total_len;
	// 			*reg_send_addr_high = (unsigned int)(((unsigned long)send_head)>>32);
	// 			*reg_send_addr_low = (unsigned int)(send_head);
	// 			*reg_send_start = (g_mutex+1)/block_num;
	// 		}
	// 		atomicAdd((int*) &g_mutex, 1);

	// 		while((g_mutex) % block_num != 0){
	// 		};
	// 		send_head += (batch_size + key_num_aligned*batch_size);
	// 	END_BLOCK_ZERO_DO

	// }
}
using namespace std;
void kvs_benchmark(param_test_t param_in){
	printf("=====start: kvs benchmark\n");
	unsigned int* device_addr_start = ((unsigned int*)param_in.map_d_ptr);
	fpga::XDMAController* controller = param_in.controller;
	unsigned long tlb_start = (unsigned long)device_addr_start;

	unsigned int* db_start = device_addr_start + size_t(16)*1024*1024*1024/4;

	unsigned long recv_start = tlb_start;
	unsigned long send_start = recv_start+size_t(8)*1024*1024*1024;
	unsigned int mac = 4;
	unsigned int ip = 0xc0a8bd0a;
	unsigned int port = 1235;
	unsigned int threads_per_block = 512;
	unsigned int batch_size = 1024;

	unsigned int block_num = 4;
	unsigned int key_num = 16;

	controller->writeReg(0,0);
	controller->writeReg(0,1);
	sleep(1);

	controller->writeReg(128,mac);
	controller->writeReg(129,ip);
	controller->writeReg(130,port);
	controller->writeReg(306,batch_size/16-1);
	controller->writeReg(291,batch_size/16-1);
	controller->writeReg(308,1024*4096/64);
	controller->writeReg(307,key_num*4);
	controller->writeReg(131,0);
	controller->writeReg(131,1);
	//tcp listen
	while (((controller->readReg(640)) >> 1) == 0)
   {
      sleep(1);
      cout << "listen status: " << controller->readReg(640) << endl;
   };
   cout << "listen status: " << controller->readReg(640) << endl;
   sleep(1);

	controller ->writeReg(304,(unsigned int)recv_start);//low
	controller ->writeReg(305,(unsigned int)(recv_start>>32));

	unsigned int* reg_recv_count		= map_reg_4(811,controller);//received package count, multiple of 16

	unsigned int* reg_send_addr_low		= map_reg_4(288,controller);
	unsigned int* reg_send_addr_high	= map_reg_4(289,controller);
	unsigned int* reg_send_len			= map_reg_4(290,controller);
	unsigned int* reg_send_start		= map_reg_4(293,controller);

	unsigned int* reg_fifo_cnt802		= map_reg_4(802,controller);
	unsigned int* reg_fifo_cnt803		= map_reg_4(803,controller);


	unsigned int* reg_button = map_reg_4(303,controller);
	controller->writeReg(303,0);

	unsigned int * atomic_count;
	int * timer;
	cudaMalloc(&atomic_count,4);
	cudaMalloc(&timer,4);
	kvs_decouple<<<block_num,threads_per_block>>>(reg_fifo_cnt802, reg_fifo_cnt803, db_start, (volatile unsigned int *)recv_start, (volatile unsigned int *)send_start, batch_size, key_num, reg_send_addr_high, reg_send_addr_low, reg_send_start, reg_send_len, reg_recv_count);
	// kvs<<<batch_size,512>>>(reg_button, db_start, atomic_count, timer, (volatile unsigned int *)recv_start, (volatile unsigned int *)send_start,batch_size, reg_send_addr_high, reg_send_addr_low, reg_send_start, reg_send_len, reg_recv_count);
	//kvs_single_block<<<1,1024>>>( (volatile unsigned int *)recv_start, (volatile unsigned int *)send_start, batch_size, reg_send_addr_high, reg_send_addr_low, reg_send_start, reg_send_len, reg_recv_count);
}