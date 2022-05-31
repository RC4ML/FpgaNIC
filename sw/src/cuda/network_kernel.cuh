#ifndef network_kernel_CUH
#define network_kernel_CUH

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "main.h"
#include "util.cuh"
#include "network.cuh"

__global__ void socket_send(socket_context_t* ctx,int* socket,int * data_addr,size_t length);

__global__ void socket_send_offload_control(socket_context_t* ctx,int* socket,int * data_addr,size_t length, int cur_idx, unsigned int* ctrl_data);

__global__ void socket_send(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length);

__device__ void _socket_send(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length);

__global__ void socket_send_pre(socket_context_t* ctx,int* socket,size_t length, size_t max_block_size);

__global__ void socket_recv_ctrl(socket_context_t* ctx,int* socket,int * data_addr,size_t length);//check

__global__ void socket_recv_ctrl(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length);//check

__device__ void _socket_recv_ctrl(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length);

__global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length);//check

__global__ void socket_recv(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length);//check

__device__ void _socket_recv(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length);

__device__ void _socket_recv_data(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length);



__global__ void send_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers,int node_type);//check

__global__ void recv_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers);//check



__global__ void socket_close(socket_context_t* ctx,int* socket);

__global__ void socket_close(socket_context_t* ctx,connection_t* connection);

__device__ void _socket_close(socket_context_t* ctx,int session_id);
#endif