# Optimizing matrix transpose operation in GPU
Matrix transpose is a very fundamental mathematical operation which is also an important **one-to-one** parallel communication pattern. In a more general sense of the term *transpose*, it is used in conversion of *Array of Structures* to *Structure of Arrays* to make similar elements contiguous in the memory. The transpose operation also has a wide variety of uses in *matrix calculus* and hence in deep learning (for example, in the backpropagation algorithm).

Here, different implementations of the matrix transpose operation are compared with a detailed account of the code. 
Since, the codes are in `CUDA C`, matrices (and other multi-dimensional arrays) are stored in *row-major order* in the linear memory
space.

This problem becomes very important to understand memory optimisations in GPU as it shows two characteristics:
1. Time is mostly spent in **fetching** data from *memory* and **storing** it back at some place else.
2. No significant **compute** operations are performed on the data, i.e., Transpose is not a compute heavy kernel like
Matrix- Multiplication or Convolution.

The following GPU based implementations are present in the code:
- Row-wise loading and column-wise storing in global memory
- Column-wise loading and row-wise storing in global memory
- Using *square* shared memory tiles with bank conflicts
- Using *square* shared memory tiles with **no** bank conflicts
- Using *rectangular* shared memory tiles with bank conflicts
- Using *rectangular* shared memory tiles with **no** bank conflicts

I also used a CPU based plain transpose implementation to get the target output to be compared with the output of the kernels.
A simple GPU based copy kernel, which just moves data from one place in global memory to another via a square shared memory, is
also used to obtain a reference time (where shared memory read and writes are bank conflict free and unlike transpose, no index swapping is involved!).

## Input and Output dimensions
- **Input-** A float matrix of dimensions `m`x`n` (`m` rows and `n` columns)
- **Output-** A float matrix of dimensions `n`x`m` (`n` rows and `m` columns) [*Obviously!*]

Note that both input and output matrices are accessed as flattened arrays (`float *`) with appropriate index calculations i.e.,
- `input[i][j]` is equivalent to `input[n*i + j]`, where `0 <= i <= m-1` and `0 <= j <= n-1` 
- `output[i][j]` is equivalent to `output[m*i + j]`, where `0 <= i <= n-1` and `0 <= j <= m-1` 

(Note that the bounds of `i` and `j` are different for input and output matrices.)

## Block and Grid dimensions
- For *square* tiles (as well as blocks), I used a 2D block of dimensions `(32, 32)` i.e., a total of 1024 threads in a block, which is also the
maximum number of *thread contexts* a block can store.
- For *rectangular* tiles (as well as blocks), `(32, 16)` dimensional blocks are used.
- Since in the most general scenario, the matrix row(`= m`) and column(`= n`) dimensions are not necessarily a multiple of `block.y` and `block.x`,
rounded up dimensions are used for the 2D grid as `(( n + block.x - 1) / block .x, (m + block.y - 1) / block .y)`. Hence there are
some *redundant threads* which do no-work and are prevented from accessing memory by array index bound-checks. *Check Fig.1 below.*

Note that blocks and grids are declared in `(x, y, z)` format whereas they are arranged in a linear fashion in (z, y, x) coordinates. Only the innermost
variable (`x`) changes for contiguous blockIds and threadIds. This becomes a matter of concern later. In the following figure, variation of block
indices y and x are shown for a 2D grid with square tiles.

<p align="center"> 
<img src = "/gridij.png">
<br>
<em>Fig. 1: Grid with actual matrix elements (gray) and redundant threads (pink portion)</em>
</p>


## Naive Row Kernel
```c++
__global__ void gpuTransposeRow(float *d_in, float *d_out, int m, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
  
    //implement out[j][i] = in[i][j]
    if(i < m && j < n)
        d_out[m*j + i] = d_in[n*i + j];
}
```

## Naive Column Kernel
```c++
__global__ void gpuTransposeCol(float *d_in, float *d_out, int m, int n)
{
    int i_o = blockIdx.x * blockDim.x + threadIdx.y;
    int j_o = blockIdx.y * blockDim.y + threadIdx.x;
    
    //implement out[i][j] = in[j][i]
    if(i_o < n && j_o < m)
    {
        d_out[m*i_o + j_o] = d_in[n*j_o + i_o];
        //printf("i = %d j = %d\n", i, j);
    }
}
```

## Square shared memory with bank conflicts
```c++
__global__ void gpuTransposeCoalesced(float *d_in, float *d_out, int m, int n)
{
    __shared__ float shared[TILE_DIM][TILE_DIM];
 
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
 
    if(i < m && j < n)
        shared[threadIdx.y][threadIdx.x] = d_in[n*i + j];
    __syncthreads();
 
    int i_o = blockIdx.x * blockDim.x + threadIdx.y;
    int j_o = blockIdx.y * blockDim.y + threadIdx.x;
  
    if(i_o < n && j_o < m)
        d_out[m*i_o + j_o] = shared[threadIdx.x][threadIdx.y];

}
```
## Square shared memory with no bank conflict
```c++
__global__ void gpuTransposeCoalescedNoBC(float *d_in, float *d_out, int m, int n)
{
    __shared__ float shared[TILE_DIM][TILE_DIM+1];
 
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
 
    if(i < m && j < n)
        shared[threadIdx.y][threadIdx.x] = d_in[n*i + j];
    __syncthreads();
 
    int i_o = blockIdx.x * blockDim.x + threadIdx.y;
    int j_o = blockIdx.y * blockDim.y + threadIdx.x;
  
    if(i_o < n && j_o < m)
        d_out[m*i_o + j_o] = shared[threadIdx.x][threadIdx.y];
}
```

## Helpful Resources
 - https://stackoverflow.com/questions/37732735/nvprof-option-for-bandwidth
