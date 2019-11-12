/*
需要设备运行库支持
nvcc -rdc=true nestedHelloWorld.cu -o nestedHelloWorld -lcudadevrt -arch=sm_35
*/

#include <stdio.h>
#include "freshman.h"

__global__ void nestedHelloworld(int const iSize, int iDepth){
    int tid = threadIdx.x;
    printf("Recursion %d, thread index %d, block %d\n", iDepth, tid, blockIdx.x);

    if (iSize == 1) return;

    int nthreads = iSize>>1;
    if (tid == 0 && nthreads > 0){
        nestedHelloworld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-----> nested execution depth %d\n", iDepth);
    }
}

int main(int argc, char** argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    dim3 block(8);
    dim3 grid(1);
    nestedHelloworld<<<grid, block>>>(8, 0);

    cudaDeviceReset();

    return 0;
}