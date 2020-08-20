//16EE10063, Swagatam Haldar, CUDA Assignment- 2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reduce_kernel(int n, int k, float *a, float *b)
{
    int blocknum, threadnum, i, aidx, bidx;
    blocknum = blockIdx.y * gridDim.x + blockIdx.x;           //Grid is 2D, sqrt(m) x sqrt(m)
    threadnum = threadIdx.x;                                  //Each block is 1D, containing 'k' threads
 
    aidx = blocknum * blockDim.x;                             //starting index of one 'k'-sized block in original array 'a'
    bidx = blocknum;                                          //since num_blocks = m = n/k = no. of elements in reduced array 'b'
    i = threadnum;                                            //i varies from 0 to k-1 for each block
 
    for(int s = 1; s < k; s *= 2)
    {
        if(i % (2*s) == 0 && i + s < k)
            a[i + aidx] += a[i + aidx + s];
        __syncthreads();
    }
    if(i == 0)
    {
        //printf("At location %d, %.2f is written by thread %d of block %d\n", bidx, a[aidx]/k, threadnum, blocknum);
        b[bidx] = a[aidx] / k;                               //average is written to appropriate location of 'b' by a single thread
    }
}

int main()
{
    int t, p, q, n, k, curlen;
    double m, sq;
    int x, y;
    float *a, *b, *da, *db;
    scanf("%d", &t);
    while(t--)
    {
        scanf("%d %d", &p, &q);
        n = pow(2, p);
        k = pow(2, q);
        m = n / k;
     
        a = (float *)malloc(n * sizeof(float));
        cudaMalloc((void **)&da, n * sizeof(float));
     
        for(int i=0; i<n; i++)
            scanf("%f", &a[i]);
     
        cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice);
        curlen = n;
     
        while(curlen >= k)
        {
            m = curlen / k;
            b = (float *)malloc(m * sizeof(float));
            cudaMalloc((void **)&db, m * sizeof(float));
         

            sq = sqrt(m);
         
            if(floor(sq) == ceil(sq))
            {
                x = y = sqrt(m);
            }
            else //If m is not a perfect square, handle it separately
            {
                x = m/2;
                y = 2;
            }

            dim3 grid(x,y,1);
            dim3 block(k,1,1);

            reduce_kernel<<<grid, block>>>(curlen, k, da, db);
         
            cudaMemcpy(b, db, m * sizeof(float), cudaMemcpyDeviceToHost);

            a = b;
            curlen = m;
         
            cudaFree(da);
            cudaFree(db);
         
            cudaMalloc((void **)&da, m * sizeof(float));
            cudaMemcpy(da, a, m * sizeof(float), cudaMemcpyHostToDevice);
        }
        for(int i=0; i<curlen; i++)
            printf("%.2f ", a[i]);
        printf("\n");
        cudaFree(da);
        free(a);
    }
    return 0;
}

/*inputs
4
3 1
1 2 3 4 5 6 7 8
4 2
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
5 3
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
6 4
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 
*/