#include <stdio.h>
#include <cuda_runtime.h>
#include "freshman.h"

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    // thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // data pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // thread id out of range
    if (idx >= n) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if (threadIdx.x % (stride*2) == 0){
            idata[threadIdx.x] += idata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){
    // thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // data pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // thread id out of range
    if (threadIdx.x >= n) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        // first data index of this thread 
        int index = 2 * idx * stride;
        // data add
        if (index < blockDim.x){
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceInterleave(int *g_idata, int *g_odata, unsigned int n){
    // thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // data pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // thread id out of range
    if (idx >= n) return;
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            idata[threadIdx.x] += idata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnroll2(int *g_idata, int *g_odata, unsigned int n){
    // thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // data pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();
    // thread id out of range
    if (idx >= n) return;
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            idata[threadIdx.x] += idata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, unsigned int n){
    // thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // data pointer of this block(s)
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling blocks
    if (idx + 7 * blockDim.x < n) {
        int el0 = g_idata[idx];
        int el1 = g_idata[idx + blockDim.x];
        int el2 = g_idata[idx + 2*blockDim.x];
        int el3 = g_idata[idx + 3*blockDim.x];
        int el4 = g_idata[idx + 4*blockDim.x];
        int el5 = g_idata[idx + 5*blockDim.x];
        int el6 = g_idata[idx + 6*blockDim.x];
        int el7 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = el0+el1+el2+el3+el4+el5+el6+el7;
    }
    __syncthreads();
    // thread id out of range
    if (idx >= n) return;
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1){
        if (threadIdx.x < stride){
            idata[threadIdx.x] += idata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // unrolling sync in blocks(stride less than 32)
    if (threadIdx.x < 32){
        volatile int *vmem = idata;
        vmem[threadIdx.x] += vmem[threadIdx.x + 32];
        vmem[threadIdx.x] += vmem[threadIdx.x + 16];
        vmem[threadIdx.x] += vmem[threadIdx.x + 8];
        vmem[threadIdx.x] += vmem[threadIdx.x + 4];
        vmem[threadIdx.x] += vmem[threadIdx.x + 2];
        vmem[threadIdx.x] += vmem[threadIdx.x + 1];
    }
    if (threadIdx.x == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, unsigned int n){
    // thread id
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    // data pointer of this block(s)
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling blocks
    if (idx + 7 * blockDim.x < n) {
        int el0 = g_idata[idx];
        int el1 = g_idata[idx + blockDim.x];
        int el2 = g_idata[idx + 2*blockDim.x];
        int el3 = g_idata[idx + 3*blockDim.x];
        int el4 = g_idata[idx + 4*blockDim.x];
        int el5 = g_idata[idx + 5*blockDim.x];
        int el6 = g_idata[idx + 6*blockDim.x];
        int el7 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = el0+el1+el2+el3+el4+el5+el6+el7;
    }
    __syncthreads();

    // unrolling in blocks
    // 这种优化需要保证blockDim.x为2的k次幂，且最大为1024
    if (blockDim.x >= 1024 && threadIdx.x < 512) idata[threadIdx.x] += idata[threadIdx.x + 512];
    __syncthreads();
    if (blockDim.x >= 512 && threadIdx.x < 256) idata[threadIdx.x] += idata[threadIdx.x + 256];
    __syncthreads();
    if (blockDim.x >= 256 && threadIdx.x < 128) idata[threadIdx.x] += idata[threadIdx.x + 128];
    __syncthreads();
    if (blockDim.x >= 128 && threadIdx.x < 64) idata[threadIdx.x] += idata[threadIdx.x + 64];
    __syncthreads();

    // unrolling sync in thread cluster(stride less than 32)
    if (threadIdx.x < 32){
        volatile int *vmem = idata;
        vmem[threadIdx.x] += vmem[threadIdx.x + 32];
        vmem[threadIdx.x] += vmem[threadIdx.x + 16];
        vmem[threadIdx.x] += vmem[threadIdx.x + 8];
        vmem[threadIdx.x] += vmem[threadIdx.x + 4];
        vmem[threadIdx.x] += vmem[threadIdx.x + 2];
        vmem[threadIdx.x] += vmem[threadIdx.x + 1];
    }
    if (threadIdx.x == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n){
    // thread id
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    // data pointer of this block(s)
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling blocks
    if (idx + 7 * blockDim.x < n) {
        int el0 = g_idata[idx];
        int el1 = g_idata[idx + blockDim.x];
        int el2 = g_idata[idx + 2*blockDim.x];
        int el3 = g_idata[idx + 3*blockDim.x];
        int el4 = g_idata[idx + 4*blockDim.x];
        int el5 = g_idata[idx + 5*blockDim.x];
        int el6 = g_idata[idx + 6*blockDim.x];
        int el7 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = el0+el1+el2+el3+el4+el5+el6+el7;
    }
    __syncthreads();

    // unrolling in blocks
    // 利用预编译减少线程束分化
    if (iBlockSize >= 1024 && threadIdx.x < 512) idata[threadIdx.x] += idata[threadIdx.x + 512];
    __syncthreads();
    if (iBlockSize >= 512 && threadIdx.x < 256) idata[threadIdx.x] += idata[threadIdx.x + 256];
    __syncthreads();
    if (iBlockSize >= 256 && threadIdx.x < 128) idata[threadIdx.x] += idata[threadIdx.x + 128];
    __syncthreads();
    if (iBlockSize >= 128 && threadIdx.x < 64) idata[threadIdx.x] += idata[threadIdx.x + 64];
    __syncthreads();

    // unrolling sync in thread cluster(stride less than 32)
    if (threadIdx.x < 32){
        volatile int *vmem = idata;
        vmem[threadIdx.x] += vmem[threadIdx.x + 32];
        vmem[threadIdx.x] += vmem[threadIdx.x + 16];
        vmem[threadIdx.x] += vmem[threadIdx.x + 8];
        vmem[threadIdx.x] += vmem[threadIdx.x + 4];
        vmem[threadIdx.x] += vmem[threadIdx.x + 2];
        vmem[threadIdx.x] += vmem[threadIdx.x + 1];
    }
    if (threadIdx.x == 0) g_odata[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, int const size){
    if (size == 1) return data[0];
    int const stride = size / 2;
    for(int i = 0; i < stride; i++){
        data[i] += data[i+stride];
    }
    if ((size % 2)!=0){
        data[stride] = data[stride*2];
        return recursiveReduce(data, stride+1);
    }
    return recursiveReduce(data, stride);
}

int main(int argc, char **argv) {
    // set up device
    initDevice(0);

    // set up data
    int size = 1<<14;

    // set up threads
    int blocksize = 512;
    if (argc > 1){
        blocksize = atoi(argv[1]);
    }
    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1)/block.x, 1);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes); 
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    initialData_int(h_idata, size);

    int *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x*sizeof(int));
    
    double iStart, iElaps;
    int gpu_sum;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    // reduceCompleteUnrollWarp8<<<grid, block>>>(d_idata, d_odata, size);
    switch (blocksize){ // 调用模板参数不能用变量
        case 1024:
            reduceCompleteUnroll<1024><<<grid, block>>>(d_idata, d_odata, size);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid, block>>>(d_idata, d_odata, size);
            break;
        case 256:
            reduceCompleteUnroll<256><<<grid, block>>>(d_idata, d_odata, size);
            break;
        case 128:
            reduceCompleteUnroll<128><<<grid, block>>>(d_idata, d_odata, size);    
            break;
    }
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
            grid.x, grid.y, block.x, block.y, iElaps);
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        printf(" %d ", h_odata[i]);
        gpu_sum += h_odata[i];
    }
    printf("reduceNeighbored %d\n", gpu_sum);

    int *tmp = (int *)malloc(bytes);
    memcpy(tmp, h_idata, bytes);
    int result = recursiveReduce(tmp, size);
    printf("%d\n", result);

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}