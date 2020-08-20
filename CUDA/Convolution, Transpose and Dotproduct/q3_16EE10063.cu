//Swagatam Haldar, 16EE10063, Question- 3
%%cuda --name q3_16EE10063.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
typedef unsigned int uint;

__global__ void gpuTransposeCoalesced1DNoBC(float *d_in, float *d_out, uint m, uint n)
{
    extern __shared__ float tile[];
 
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
 
    if(i < m && j < n)
        tile[(TILE_DIM+1)*threadIdx.y + threadIdx.x] = d_in[n*i + j];
    __syncthreads();
 
    uint i_o = blockIdx.x * blockDim.x + threadIdx.y;
    uint j_o = blockIdx.y * blockDim.y + threadIdx.x;
  
    if(i_o < n && j_o < m)
        d_out[m*i_o + j_o] = tile[(TILE_DIM+1)*threadIdx.x + threadIdx.y];
}

void print_matrix(float *a, uint n)
{
    for(uint i=0; i<n; i++)
    {
        for(uint j=0; j<n; j++)
            printf("%.2f ", a[n*i + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    uint t, n;
    float *h_in, *h_out, *d_in, *d_out;
    
    scanf("%d", &t);
    
    while(t--)
    {
        scanf("%d", &n);

        /* Allocate host and device placeholders */
        h_in = (float *)malloc(n * n * sizeof(float));
        h_out = (float *)malloc(n * n * sizeof(float));
        cudaMalloc((void **)&d_in, n * n * sizeof(float));
        cudaMalloc((void **)&d_out, n * n * sizeof(float));
     
        /* Read input and copy to device */
        for(uint i=0; i<n*n; i++)
            scanf("%f", &h_in[i]);
        cudaMemcpy(d_in, h_in, n * n *sizeof(float), cudaMemcpyHostToDevice);
     
        /* Launch parameters and kernel launch */
        dim3 block (TILE_DIM, TILE_DIM);
        dim3 grid ((n + block.x - 1) / block .x, (n + block.y - 1) / block .y);
     
        gpuTransposeCoalesced1DNoBC<<<grid, block, TILE_DIM * (TILE_DIM+1) * sizeof(float)>>>(d_in, d_out, n, n);
     
        /* Get the result back to host and print */
        cudaMemcpy(h_out, d_out, n * n *sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(h_out, n);
     
        /* Free the place holders for next test case */
        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
    }
 
    return 0;
}