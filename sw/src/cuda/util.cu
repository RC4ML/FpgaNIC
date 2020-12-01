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