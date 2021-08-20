#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "network.cuh"

typedef struct param_cuda_thread{
	volatile unsigned int* devReadCountAddr0;
	volatile unsigned int* devReadCountAddr1;
	volatile unsigned int* devReadCountAddr2;
	volatile unsigned int* devReadCountAddr3;

	volatile unsigned int* devWriteCountAddr0;
	volatile unsigned int* devWriteCountAddr1;
	volatile unsigned int* devWriteCountAddr2;
	volatile unsigned int* devWriteCountAddr3;

	unsigned int* devVAddr0;
	unsigned int* devVAddr1;
	unsigned int* devVAddr2;
	unsigned int* devVAddr3;

	unsigned int* dstAddr0;
	unsigned int* dstAddr1;
	unsigned int* dstAddr2;
	unsigned int* dstAddr3;

	int threadsPerBlock;
	int blocks;
	uint64_t data_length;
	int offset;
	int buffer_pages;
}param_cuda_thread_t;


 __global__ void movThread(param_cuda_thread_t param);
 //__global__ void writeBypassReg(volatile unsigned int *dev_addr,int *blocks);
//  __global__ void readBypassReg(volatile unsigned int *dev_addr,int *blocks);
 __global__ void writeReg(volatile unsigned int *dev_addr,int *blocks);
 __global__ void compute(int * data,size_t length,int offset);
 __global__ void verify(int * data,size_t length,int offset);

#endif