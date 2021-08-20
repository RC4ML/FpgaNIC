#ifndef network_CUH
#define network_CUH
#include <stdio.h>
#include <stdint.h>
#include "util.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "main.h"
#include "network_kernel.cuh"








socket_context_t* get_socket_context(unsigned int *dev_buffer,unsigned int *tlb_start_addr,fpga::XDMAController* controller,int node_type);//check

__global__ void create_socket(socket_context_t* ctx,int* socket);//check

__global__ void socket_listen(socket_context_t* ctx,int *socket, int port);//check

// __global__ void socket_send(socket_context_t* ctx,int* socket,int * data_addr,size_t length,sock_addr_t dst_addr);//check

// __global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length);//check


__global__  void connect(socket_context_t* ctx,int* socket,sock_addr_t addr);//check

__global__ void accept(socket_context_t* ctx,int* socket,connection_t* connection);

unsigned int* map_reg_4(int reg,fpga::XDMAController* controller);//check

unsigned int* map_reg_64(int reg,fpga::XDMAController* controller);

unsigned int* map_reg_cj(int reg,fpga::XDMAController* controller,size_t length);

__device__ void move_data_to_send_buffer(socket_context_t* ctx,int buffer_id,int block_length,int *data_addr);//check

__device__ void move_data_from_recv_buffer(socket_context_t* ctx,int buffer_id,int block_length,int *data_addr);//check

__device__ connection_t get_session(socket_context_t* ctx,int socket,sock_addr_t dst_addr);//check

__device__ int get_session_first(socket_context_t* ctx,int socket);//check

__device__ bool check_socket_validation(socket_context_t* ctx,int socket);//check

__device__ int enroll(socket_context_t* ctx,int socket_id,int *data_addr,size_t length);//check

__device__ void write_bypass(volatile unsigned int *dev_addr,unsigned int *data);

__device__ int read_info(socket_context_t* ctx);

__device__ int get_empty_buffer(socket_context_t* ctx);

__device__ int fetch_head(socket_context_t* ctx,int buffer_id);
#endif