#ifndef INFERENCE_CUH
#define INFERENCE_CUH

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "main.h"
#include "tool/log.hpp"

void inference_sample(param_test_t param_in);
#endif