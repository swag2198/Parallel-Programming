//Swagatam Haldar, 16EE10063, Question- 1
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv2Dbasic(float *d_N, float *d_P, int n, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = m/2;
 
    int startrow = i-k;
    int startcol = j-k;
    int row, col;
    float psum = 0;

    if(i < n && j < n)
    {
        for(int r = 0; r < m; r++)
        {
            for(int c = 0; c < m; c++)
            {
                row = startrow + r;
                col = startcol + c;
                if(row >= 0 && col >= 0 && row < n && col < n)
                    psum += d_N[n*row + col];
                
            }
        }
        d_P[n*i + j] = psum / float(m*m);
    }
}

void print_matrix(float *a, int n)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
            printf("%.2f ", a[n*i + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    int t, n;
    float *h_N, *h_P, *d_N, *d_P;
 
    scanf("%d", &t);
    while(t--)
    {
        scanf("%d", &n);
     
        /* Allocate cpu and gpu placeholders */
        h_N = (float *)malloc(n * n * sizeof(float));
        h_P = (float *)malloc(n * n * sizeof(float));
        cudaMalloc((void **)&d_N, n * n * sizeof(float));
        cudaMalloc((void **)&d_P, n * n * sizeof(float));
     
        /* Read input and copy to device */
        for(int i=0; i<n*n; i++)
            scanf("%f", &h_N[i]);
        cudaMemcpy(d_N, h_N, n * n *sizeof(float), cudaMemcpyHostToDevice);
     
        /* Kernel launch */
        dim3 block(32,32,1);
        float grid_dim = ceil(sqrt(n*n / 1024.0));
        dim3 grid(int(grid_dim), int(grid_dim), 1);
        conv2Dbasic<<<grid, block>>>(d_N, d_P, n, 3);
     
        /* Get the output to host and print it */
        cudaMemcpy(h_P, d_P, n * n *sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(h_P, n);
     
        /* Free memory */
        cudaFree(d_N);
        cudaFree(d_P);
        free(h_P);
        free(h_N);
    }
    
    return 0;
}