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
#define MAX_BUFFER_NUM 4
#define SINGLE_BUFFER_LENGTH 25*1024*1024
#define TOTAL_BUFFER_LENGTH 100*1024*1024

#define INFO_BUFFER_LENGTH 2*1024*1024
#define MAX_PACKAGE_LENGTH 2*1024*1024

#define FLOW_CONTROL_RATIO 0.5
#define MAX_ACCEPT_LIST_LENGTH 64
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
	volatile unsigned int* con_session_status;

	volatile unsigned int* listen_status;
	volatile unsigned int* listen_port;
	volatile unsigned int* listen_start;

	volatile unsigned int* conn_ip;
	volatile unsigned int* conn_port;
	volatile unsigned int* conn_buffer_id;
	volatile unsigned int* conn_start;

	volatile unsigned int* conn_response;
	volatile unsigned int* conn_res_start;
	volatile unsigned int* conn_re_session_id;


	//bypass
	volatile unsigned int* send_data_cmd_bypass_reg;
	volatile unsigned int* recv_read_count_bypass_reg;
}fpga_registers_t;



typedef struct connection_node{
	int session_id;
	int src_ip;//to print, no other use
	int src_port;//to print, no other use
	int buffer_id;
	bool valid;
}connection_t;



typedef struct socket_type{
	int buffer_id;
	int valid;
}socket_type_t;

typedef struct buffer_type{
	int valid;
	int session_id;
	int socket_id;
	int type;//0 for socket   1 for connection
	connection_t* connection;//only for type 1
}buffer_type_t;


typedef struct socket_context{
	volatile unsigned long send_read_count[MAX_BUFFER_NUM];
	volatile unsigned long send_write_count[MAX_BUFFER_NUM];
	volatile unsigned long recv_read_count[MAX_BUFFER_NUM];

	bool buffer_valid[MAX_BUFFER_NUM];
	int send_buffer_offset[MAX_BUFFER_NUM];
	int recv_buffer_offset[MAX_BUFFER_NUM];
	buffer_type_t buffer_info[MAX_BUFFER_NUM];
	int buffer_read_count_record[MAX_BUFFER_NUM];//todo init=1

	int buffer_used;
	int info_offset;
	int info_count;

	volatile unsigned int* send_buffer;//check
	volatile unsigned int* recv_buffer;//check
	volatile unsigned int* info_buffer;//check
	int socket_num;//check
	socket_type_t	socket_info[1024];

	int is_listening;
	int server_port;
	int server_socket_id;
	int is_accepting;
	int accepted;
	connection_t * connection_builder;
	connection_t connection_tbl[1024];
	
	int mutex;//check

	//registers write

	//registers read
	volatile unsigned int* con_session_status;//check

	volatile unsigned int* listen_status;
	volatile unsigned int* listen_port;
	volatile unsigned int* listen_start;

	volatile unsigned int* conn_ip;
	volatile unsigned int* conn_port;
	volatile unsigned int* conn_buffer_id;
	volatile unsigned int* conn_start;

	volatile unsigned int* conn_response;
	volatile unsigned int* conn_res_start;
	volatile unsigned int* conn_re_session_id;



	//bypass
	volatile unsigned int* send_data_cmd_bypass_reg;
	volatile unsigned int* recv_read_count_bypass_reg;
}socket_context_t;

#endif