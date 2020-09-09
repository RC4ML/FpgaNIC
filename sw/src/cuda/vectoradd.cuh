#ifndef VECTORADD_CUH
#define VECTORADD_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "main.h"
extern "C"
void useCUDA();
void write_bypass(void* addr);
void read_bypass(void* addr);
void data_mover(param_mover_t param_mover);
void test_mover(void* write_count_addr,void* read_count_addr,void* v_addr);

typedef struct param_mov_thread{
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
}param_mov_thread_t;

#define ErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif