#ifndef HLL_CUH
#define HLL_CUH
#include "main.h"
#include <cuda.h>
#include <cuda_runtime.h>
void hll_sample(param_test_t param_in);
__device__ unsigned long MurMurHash3(unsigned int * key);
#endif