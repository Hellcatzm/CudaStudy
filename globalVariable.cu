#include<cuda_runtime.h>
#include<stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable(){
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0f;
}

int main(){
    float value = 3.14f;
    // cudaMemcpyToSymbol(devData, &value, sizeof(float));
    float *dptr = NULL;
    cudaGetSymbolAddress((void**)&dptr, devData);  // Symbol不是地址,本句获取其全局地址
    cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice);
    printf("Host:   copied %f to the global variable\n", value);

    checkGlobalVariable<<<1, 1>>>();

    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host:   the value changed by the kernel to %f\n", value);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}