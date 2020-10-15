#ifndef __INPUT_HPP__
#define __INPUT_HPP__
#include <iostream>
#include <string>
#include "fpga/XDMA.h"
#include "fpga/XDMAController.h"
using namespace std;
void start_cmd_control(fpga::XDMAController* controller);
#endif