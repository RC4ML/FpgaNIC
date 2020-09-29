#ifndef network_CUH
#define network_CUH
#include <stdio.h>
#include <stdint.h>
#include "util.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "main.h"
#include "network_kernel.cuh"








socket_context_t* get_socket_context(unsigned int *dev_buffer,unsigned int *tlb_start_addr,fpga::XDMAController* controller);//check

__global__ void create_socket(socket_context_t* ctx,int* socket);//check

__global__ void socket_listen(socket_context_t* ctx,int socket, int port);//check

// __global__ void socket_send(socket_context_t* ctx,int* socket,int * data_addr,size_t length,sock_addr_t dst_addr);//check

// __global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length);//check


__device__  int connect(socket_context_t* ctx,int socket,sock_addr_t addr);//check

unsigned int* map_reg_4(int reg,fpga::XDMAController* controller);//check

__device__ unsigned int get_info_tbl_index(socket_context_t* ctx);//check

__device__ void move_data(socket_context_t* ctx,int block_length,int *data_addr,int addr_offset);//check

__device__ void move_data_recv(socket_context_t* ctx,int block_length,int *data_addr,int addr_offset);//check

__device__ int get_session(socket_context_t* ctx,int socket,sock_addr_t dst_addr);//check

__device__ int get_session_first(socket_context_t* ctx,int socket);//check

__device__ bool check_socket_validation(socket_context_t* ctx,int socket);//check

__device__ recv_info_t read_info(socket_context_t* ctx,int index);//check

__device__ int enroll(socket_context_t* ctx,int socket_id,int *data_addr,size_t length);//check

#endif