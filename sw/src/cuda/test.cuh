#ifndef _TEST_CUH_
#define _TEST_CUH_
#include "main.h"

void test_latency_fpga_cpu(param_test_t param_in);
void test_latency_fpga_gpu(param_test_t param_in);
void test_cpu_gpu(param_test_t param_in);
void test_simple(int stride);
void test_gpu_throughput(param_test_t param_in);
void cj_debug(param_test_t param_in);
void test_2080(param_test_t param_in);
#endif