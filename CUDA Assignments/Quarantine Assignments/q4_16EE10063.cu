//Swagatam Haldar, 16EE10063, Question- 4
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024

/* Note that it is expected that input vector lengths will be > 1024 and powers of 2 */

using namespace std;
typedef unsigned int ui;

void print_array(float *A, ui n)
{
    cout<<"+++ Printing float array of length = "<<n<<endl;
    cout<<"    ";
    for(ui i = 0; i < n; i++)
        printf("%.2f ", A[i]);
    cout<<endl;
}

__global__ void dotproduct(float *d_A, float *d_B, float *d_C, int n)
{
    __shared__ float tile[BLOCK_DIM];
    ui i = blockIdx.x * blockDim.x + threadIdx.x;
    ui tid = threadIdx.x;
 
    tile[tid] = (i < n) ? (d_A[i] * d_B[i]) : 0;
    __syncthreads();
 
    for(ui s = blockDim.x/2; s > 32; s >>= 1)
    {
        if(tid < s)
            tile[tid] += tile[tid + s];
        __syncthreads();
    }
 
    if(tid < 32)
    {
    	/* In colab the code was not working properly without the __syncthreads() on Pascal GPU */
        tile[tid] += tile[tid + 32];
        __syncthreads();
        tile[tid] += tile[tid + 16];
        __syncthreads();
        tile[tid] += tile[tid + 8];
        __syncthreads();
        tile[tid] += tile[tid + 4];
        __syncthreads();
        tile[tid] += tile[tid + 2];
        __syncthreads();
        tile[tid] += tile[tid + 1];
        __syncthreads();
    }

    if(tid == 0)
        d_C[blockIdx.x] = tile[0];
}

__global__ void reduce(float *d_B, float *d_C, int k)
{
    __shared__ float tile[BLOCK_DIM];
    ui i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    ui tid = threadIdx.x;

    tile[tid] = d_B[i] + d_B[i + BLOCK_DIM];
    __syncthreads();
 
    for(ui s = blockDim.x/2; s > 32; s >>= 1)
    {
        if(tid < s)
            tile[tid] += tile[tid + s];
        __syncthreads();
    }
 
    if(tid < 32)
    {
        tile[tid] += tile[tid + 32];
        __syncthreads();
        tile[tid] += tile[tid + 16];
        __syncthreads();
        tile[tid] += tile[tid + 8];
        __syncthreads();
        tile[tid] += tile[tid + 4];
        __syncthreads();
        tile[tid] += tile[tid + 2];
        __syncthreads();
        tile[tid] += tile[tid + 1];
        __syncthreads();
    }
 
    if(tid == 0)
    {
        d_C[blockIdx.x] = tile[0];
    }
}

void cpuReduce(float *a, ui n)
{
    float sum = 0.0;
    for(ui i=0; i<n; i++)
        sum += a[i];
 
    a[0] = sum;
}

int main(int argc, char* argv[])
{
    ui t, n, n1, curlen;
    size_t size, size1;
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C; //for gpu computations
 
    scanf("%d", &t);
    //t = 3;
 
    while(t--)
    {
        scanf("%d", &n);
        //n = 33554432;

        /* Allocate host and device memories */
        size = n * sizeof(float);
        n1 = (n + BLOCK_DIM - 1) / BLOCK_DIM;
        size1 = n1 * sizeof(float);

        h_A = (float *)malloc(size);
        h_B = (float *)malloc(size);
        h_C = (float *)malloc(size1);
        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C, size1);
     
        /* Read inputs and copy to device */
        for(uint i=0; i<n; i++)
        {
            scanf("%f", &h_A[i]);
            //h_A[i] = 1;
        }
            
        for(uint i=0; i<n; i++)
        {
            scanf("%f", &h_B[i]);
            //h_B[i] = 1;
        }
        cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);
     
        /* Dot product kernel launch for getting the partial sum array */
        dim3 block(BLOCK_DIM, 1, 1);
        dim3 grid((n + block.x - 1)/block.x, 1, 1);
        dotproduct<<<grid, block>>>(d_A, d_B, d_C, n);
        cudaMemcpy(h_C, d_C, size1, cudaMemcpyDeviceToHost);
        //print_array(h_C, n1);
     
        /* Call reduce kernel in a loop until size reduces to less than BLOCK_DIM */
        curlen = n1;
        while(curlen > BLOCK_DIM)
        {
            n1 = curlen;
            grid.x = (curlen + BLOCK_DIM - 1) / BLOCK_DIM;
            grid.x = grid.x / 2;
        
            cudaFree(d_B);
            cudaMalloc((void **)&d_B, grid.x * sizeof(float));
            free(h_C);
            h_C = (float *)malloc(grid.x * sizeof(float));

            reduce<<<grid, block>>>(d_C, d_B, n1);
            cudaMemcpy(h_C, d_B, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
        
            cudaFree(d_C);
            cudaMalloc((void **)&d_C, grid.x * sizeof(float));
            cudaMemcpy(d_C, h_C, grid.x * sizeof(float), cudaMemcpyHostToDevice);
        
            curlen = grid.x;
            //printf("curlen = %d\n", curlen);
            //print_array(h_C, curlen); 
        }

        cpuReduce(h_C, curlen);
        printf("%.2f\n", h_C[0]);
    }
    return 0;
}
