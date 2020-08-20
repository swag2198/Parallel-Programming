//16EE10063, Swagatam Haldar, CUDA Assignment- 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void process_kernel1(float *input1, float *input2, float *output, int n)
{
    int blocknum, threadnum, idx; //1D Data to 3D thread grid of 2D blocks
    blocknum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    //threadnum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    threadnum = threadIdx.y * blockDim.x + threadIdx.x;
    //idx = blocknum * (blockDim.x * blockDim.y * blockDim.z) + threadnum;
    idx = blocknum * (blockDim.x * blockDim.y) + threadnum;
 
    if(idx < n)
    {
        output[idx] = sin(input1[idx]) + cos(input2[idx]);
    }
}

__global__ void process_kernel2(float *input, float *output, int n)
{
    int blocknum, threadnum, idx; //1D Data to 2D thread packing grid of 3D blocks
    blocknum = blockIdx.y * gridDim.x + blockIdx.x;
    threadnum = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    idx = blocknum * (blockDim.x * blockDim.y * blockDim.z) + threadnum;
  
    if(idx < n)
    {
        output[idx] = log(input[idx]);
    }
}

__global__ void process_kernel3(float *input, float *output, int n)
{
    int blocknum, threadnum, idx; //1D Data to 1D thread packing grid of 2D blocks
    blocknum = blockIdx.x;
    threadnum = threadIdx.y * blockDim.x + threadIdx.x;
    idx = blocknum * (blockDim.x * blockDim.y) + threadnum;
 
    if(idx < n)
    {
        output[idx] = sqrt(input[idx]);
    }
}

int main()
{
    int n = 16384; /* 4*2*2*32*32*1 */
    /*printf("n = ");
    scanf("%d", &n);*/
    float *input1, *input2, *output;
    input1 = (float *)malloc(n*sizeof(float));
    input2 = (float *)malloc(n*sizeof(float));
    output = (float *)malloc(n*sizeof(float));

    for(int i=0; i<n; i++)
        scanf("%f", &input1[i]);

    for(int i=0; i<n; i++)
        scanf("%f", &input2[i]);
    
    float *d1 = NULL;
    float *d2 = NULL;
    float *dout1 = NULL;
    float *dout2 = NULL;
    float *dout3 = NULL;
 
    cudaMalloc((void **)&d1, n*sizeof(float));
    cudaMalloc((void **)&d2, n*sizeof(float));
 
    cudaMalloc((void **)&dout1, n*sizeof(float));
    cudaMalloc((void **)&dout2, n*sizeof(float));
    cudaMalloc((void **)&dout3, n*sizeof(float));
 
    cudaMemcpy(d1, input1, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d2, input2, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid1(4,2,2);
    dim3 block1(32,32,1);
 
    dim3 grid2(2,8,1);
    dim3 block2(8,8,16);
 
    dim3 grid3(16,1,1);
    dim3 block3(128,8,1);
 
    process_kernel1<<<grid1, block1>>>(d1, d2, dout1, n);
    process_kernel2<<<grid2, block2>>>(dout1, dout2, n);
    process_kernel3<<<grid3, block3>>>(dout2, dout3, n);
 
    cudaMemcpy(output, dout3, n*sizeof(float), cudaMemcpyDeviceToHost);
 
    for(int i=0; i<n; i++)
        printf("%.2f ", output[i]);
    printf("\n");
 
    return 0;
}