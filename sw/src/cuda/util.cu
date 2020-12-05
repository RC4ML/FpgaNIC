#include "util.cuh"
#include "tool/log.hpp"

__device__ int is_zero(volatile unsigned int* reg,int bit){
	return !((*reg)&(1<<bit));
}

__device__ int wait_done(volatile unsigned int* reg,int bit){
	cjdebug("wait done bit:%d\n",bit);
	size_t time_out = 10000000000;
	size_t s;
	s = clock64();
	while(is_zero(reg,bit)){
		if(clock64()-s>time_out){
			cjdebug("waited reg:%x\n",*reg);
			return -1;
		}
	}
	return (int)*reg;
}

__device__ void lock(int *mutex){
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex){
	atomicExch(mutex, 0);
}

__device__ int cu_sleep(double seconds){
	size_t s = clock64();
	while( clock64()-s<size_t(3000000000.0*seconds)){//1s = 3000000000
		// if(seconds==0.01){
		// 	printf("here is the %ld %ld\n",s,clock64());
		// }
	};
	return int(s);
}

unsigned int get_ip()
{
    int                 sockfd;
    struct sockaddr_in  sin;
    struct ifreq        ifr;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        perror("socket error");
        exit(1);
    }
    strncpy(ifr.ifr_name, ETH_NAME, IFNAMSIZ);      //Interface name

    if (ioctl(sockfd, SIOCGIFADDR, &ifr) == 0) {    //SIOCGIFADDR 获取interface address
        memcpy(&sin, &ifr.ifr_addr, sizeof(ifr.ifr_addr));
		std::string ip =  inet_ntoa(sin.sin_addr);
		cjinfo("ip: %s: ", ip.c_str());
		unsigned int ip_int = (unsigned int)htonl(inet_addr(ip.data()));
		cjinfo("ip:%x\n",ip_int);
		return ip_int;
		
    }else{
		cjerror("get ip failed!\n");
		return 0;
	}
}

double get_fre(){
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);
	cjinfo("GPU最大时钟频率: %.0f MHz (%0.2f GHz)\n",device_prop.clockRate*1e-3f, device_prop.clockRate*1e-6f);
	return device_prop.clockRate*1e-6f;//Ghz
	
}

__global__ void test_timer_device()
{
	clock_t s;
	BEGIN_SINGLE_THREAD_DO
		s = clock64();
		while(clock64()-s<17700000000){//10s

		}
	END_SINGLE_THREAD_DO
    
}
void test_timer(){
	test_timer_device<<<1,1>>>();
	sleep(20);
}

__device__ unsigned long MurMurHash3(unsigned int * key){
	unsigned int data = *key;
	unsigned long seed = 0xbaadf00d;
	unsigned long len = 4;

	unsigned long h1 = seed;
	unsigned long h2 = seed;
	const unsigned long c1 = 0x87c37b91114253d5;
	const unsigned long c2 = 0x4cf5ad432745937f;
	const unsigned long c3 = 0xff51afd7ed558ccd;
	const unsigned long c4 = 0xc4ceb9fe1a85ec53;

	unsigned long k1 = 0;

	unsigned long  t0 =  (unsigned char) data;
	unsigned long  t1 =  (unsigned char)(data >> 8);
	unsigned long  t2 =  (unsigned char)(data >> 16);
	unsigned long  t3 =  (unsigned char)(data >> 24);
	k1 ^= t3 << 24;
	k1 ^= t2 << 16;
	k1 ^= t1 << 8;
	k1 ^= t0 << 0;
	k1 *= c1;
	k1 = (k1 << 31) | (k1 >> (64 - 31));
	k1 *= c2;
	h1 ^= k1;

	h1 ^= len;
	h2 ^= len;

	h1 += h2;
	h2 += h1;

	h1 ^= h1 >> 33;
	h1 *= c3;
	h1 ^= h1 >> 33;
	h1 *= c4;
	h1 ^= h1 >> 33;

	h2 ^= h2 >> 33;
	h2 *= c3;
	h2 ^= h2 >> 33;
	h2 *= c4;
	h2 ^= h2 >> 33;

	h1 += h2;
	h2 += h1;
	return h1;
}

__device__ void hll(volatile unsigned int *data,int *mem){
	unsigned long hash_res = MurMurHash3((unsigned int*)data);

	//printf("data:%d hash_res:%lx\n",data[0],hash_res);
	
	int bucket_id = hash_res>>(64-16);//2^16 buckets
	int position = __ffsll(hash_res & 0xffffff);
	if(position==0){
		position = 64-1;
	}
	//printf("bucket:%d pos:%d\n",bucket_id,position);
	atomicMax(mem+bucket_id,position);
}
__global__ void hll_test(int *data,int *mem){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	BEGIN_SINGLE_THREAD_DO
		for(int i=0;i<1024;i++){
			data[i]=i;
		}
		for(int i=0;i<65536;i++){
			mem[i]=0;
		}
	END_SINGLE_THREAD_DO
	hll((volatile unsigned int *)data+index,mem);
}