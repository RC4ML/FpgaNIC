#ifndef __TEST_HPP__
#define __TEST_HPP__
#include "main.h"
void test_throughput(param_test_t param);
void print_speed(param_test_t param,int length,int mode);
void stream_transfer(param_test_t param);
void socket_send_test(param_test_t param_in);
void socket_send_test_offload_control(param_test_t param_in);
#endif