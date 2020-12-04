#ifndef INFERENCE_UTIL_CUH
#define INFERENCE_UTIL_CUH
void load_data(float data[][150528]);

__global__ void get_max(float* data);
#endif