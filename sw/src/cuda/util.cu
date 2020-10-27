#include "util.cuh"


__device__ int is_zero(volatile unsigned int* reg,int bit){
	return !((*reg)&(1<<bit));
}

__device__ int wait_done(volatile unsigned int* reg,int bit){
	printf("wait done bit:%d\n",bit);
	size_t time_out = 50000000000;
	size_t s;
	s = clock64();
	while(is_zero(reg,bit)){
		if(clock64()-s>time_out){
			printf("waited reg:%x\n",*reg);
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

__device__ int cu_sleep(int seconds){
	size_t s = clock64();
	while( clock64()-s<size_t(1000000000)*seconds){
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
		printf("ip: %s: ", ip.c_str());
		unsigned int ip_int = (unsigned int)htonl(inet_addr(ip.data()));
		printf("ip:%x\n",ip_int);
		return ip_int;
		
    }else{
		printf("get ip failed!\n");
		return 0;
	}
}