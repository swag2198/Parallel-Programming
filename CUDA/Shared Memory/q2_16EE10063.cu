//Swagatam Haldar, 16EE10063, Question- 2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define W 16
typedef unsigned int uint;

__global__ void gpuTransposeRectNoBC(float *d_in, float *d_out, uint m, uint n)
{
    __shared__ float rect[W][2*W+1];
 
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
 
 
    if(i < m && j < n)
        rect[threadIdx.y][threadIdx.x] = d_in[n*i + j];
    __syncthreads();
 
    uint threadnum = threadIdx.y * blockDim.x + threadIdx.x;
    uint tidy_o = threadnum / blockDim.y;
    uint tidx_o = threadnum % blockDim.y;
 
    uint i_o = blockIdx.x * blockDim.x + tidy_o;
    uint j_o = blockIdx.y * blockDim.y + tidx_o;
 
    if(i_o < n && j_o < m)
        d_out[m*i_o + j_o] = rect[tidx_o][tidy_o];
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
        dim3 blockr (32, 16);
        dim3 gridr ((n + blockr.x - 1)/blockr.x, (n + blockr.y - 1)/blockr.y);
     
        gpuTransposeRectNoBC<<<gridr, blockr>>>(d_in, d_out, n, n);
     
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