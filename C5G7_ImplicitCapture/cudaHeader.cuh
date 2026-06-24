#pragma once

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
//#include <curand.h>
#include "Constants.cuh"
#include "RNG.cuh"
//#include "XSParser.cuh"

#ifdef __CUDACC__
	#define HD __host__ __device__
	#define H  __host__
	#define D  __device__
	#define G  __global__
#else
	#define HD
	#define H
	#define D
	#define G
#endif

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); \
  } \
} while(0)

#define CUDA_KERNEL_CHECK() do { \
  CUDA_CHECK(cudaGetLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)