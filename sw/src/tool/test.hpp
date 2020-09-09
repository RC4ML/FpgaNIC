#ifndef __TEST_HPP__
#define __TEST_HPP__
#include "main.h"
void test_throughput(param_test_t param);
void print_speed(param_test_t param,int length,int mode);
void control_reg(param_test_t param);
void bypass_reg(param_test_t param);
void stream_transfer(param_test_t param);
#endif