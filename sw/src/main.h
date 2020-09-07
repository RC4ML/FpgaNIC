#ifndef __MAIN_H__
#define __MAIN_H__

#include "fpga/XDMA.h"
#include "fpga/XDMAController.h"

void set_page_table();
void copy_to_cpu();
void close_device();
typedef struct param_test{
	fpga::XDMAController* controller;
	uint64_t addr;
	uint32_t *cpu_buf;
}param_test_t;
#endif