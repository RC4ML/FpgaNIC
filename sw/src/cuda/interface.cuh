#ifndef interface_CUH
#define interface_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "main.h"
extern "C"


void write_bypass(void* addr);
void read_bypass(void* addr);
void data_mover(param_mover_t param_mover);
void socket_sample(param_interface_socket_t param_in);
void socket_sample_offload_control(param_interface_socket_t param_in);
void pressure_test(param_test_t param_in,int burst,int ops,int start);


#endif