#ifndef util_CUH
#define util_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "main.h"
#include <net/if.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <string>

#define MAX_CMD 50
#define MAX_BLOCK_SIZE 2*1024*1024
#define MAX_INFO_NUM 1024

#define ETH_NAME    "eno1"

#define FIRST_THREAD_IN_BLOCK() ((threadIdx.x + threadIdx.y + threadIdx.z) == 0)
#define FIRST_BLOCK() ( blockIdx.x + blockIdx.y + blockIdx.z == 0)
#define BEGIN_DO_AS_BLOCK __syncthreads(); if(FIRST_THREAD_IN_BLOCK()) { do{
#define END_DO_AS_BLOCK }while(0); } __syncthreads();

#define BEGIN_SINGLE_THREAD_DO __syncthreads(); if(FIRST_BLOCK()&&FIRST_THREAD_IN_BLOCK()) { do{
#define END_SINGLE_THREAD_DO }while(0); } __syncthreads();

#define ErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
unsigned int get_ip();

__device__ int is_zero(volatile unsigned int* reg,int bit);

__device__ int wait_done(volatile unsigned int* reg,int bit);

__device__ void lock(int *mutex);

__device__ void unlock(int *mutex);

__device__ int cu_sleep(int seconds);

typedef	struct sock_addr{
		int ip;
		int mac;
		int port;
}sock_addr_t;

typedef struct fpga_registers{
	volatile unsigned int * read_count;
	volatile unsigned int* send_write_count;
	volatile unsigned int* recv_read_count;

	volatile unsigned int* con_session_status;
	volatile unsigned int* send_read_count;
	volatile unsigned int* recv_write_count;
	volatile unsigned int* listen_status;
	volatile unsigned int* listen_port;
	volatile unsigned int* listen_start;

	volatile unsigned int* conn_ip;
	volatile unsigned int* conn_port;
	volatile unsigned int* conn_start;

	volatile unsigned int* send_info_session_id;
	volatile unsigned int* send_info_addr_offset;
	volatile unsigned int* send_info_length;
	volatile unsigned int* send_info_start;
}fpga_registers_t;

typedef struct send_info{
	unsigned int addr_offset;
	unsigned int length;
	int session_id;
	int valid;
}send_info_t;


typedef struct recv_info{
	unsigned int ip;
	unsigned int src_port;
	unsigned int dst_port;
	unsigned int length;
	int session_id;
	int session_close;
	unsigned int addr_offset;
}recv_info_t;

typedef struct session_node{
	unsigned int session_id;
	int ip;
	int port;
	bool valid;
}session_node_t;

typedef struct enroll_node{
	int socket_id;
	int session_id;
	int port;
	int type;
	int done;
	int *data_addr;
	size_t length;
	size_t cur_length;
}enroll_node_t;

typedef struct socket_type{
	int type;
	int port;
	unsigned int session_id;
}socket_type_t;

typedef struct socket_context{
	volatile unsigned int* send_buffer;//check
	volatile unsigned int* recv_buffer;//check
	volatile unsigned int* info_buffer;//check
	int socket_num;//check
	socket_type_t	socket_info[1024];
	session_node_t session_tbl[1024][256];
	send_info_t send_info_tbl[MAX_CMD];
	enroll_node_t enroll_list[128];
	int enroll_list_pointer;//check
	unsigned int send_info_tbl_index;//check
	unsigned int send_info_tbl_pointer;//check
	int mutex;//check
	int current_send_addr_offset;//check
	//registers write
	volatile unsigned int* read_count;//check

	volatile unsigned int* send_write_count;//check

	volatile unsigned int* recv_read_count;//check

	//registers read
	volatile unsigned int* con_session_status;//check

	volatile unsigned int* listen_status;
	volatile unsigned int* listen_port;
	volatile unsigned int* listen_start;

	volatile unsigned int* conn_ip;
	volatile unsigned int* conn_port;
	volatile unsigned int* conn_start;

	volatile unsigned int* send_info_session_id;
	volatile unsigned int* send_info_addr_offset;
	volatile unsigned int* send_info_length;
	volatile unsigned int* send_info_start;

	volatile unsigned int* send_read_count;//check

	volatile unsigned int* recv_write_count;//check
}socket_context_t;

#endif