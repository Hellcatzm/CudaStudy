#include <stdio.h>
#include "freshman.h"

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int Nx, int Ny){
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * Ny + ix;

    if (ix < Nx && iy < Ny){
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx * ny;
    int nBytes = nxy*sizeof(float);

    // Malloc
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    memset(gpuRef, 0, nBytes);

    // Cuda Malloc
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    if (argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1)/block.x, (ny - block.y - 1)/block.y);

    double iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
            grid.x, grid.y, block.x, block.y, iElaps);

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // free device
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // free host
    free(h_A);
    free(h_B);
    free(gpuRef);

    cudaDeviceReset();

    return 0;
}