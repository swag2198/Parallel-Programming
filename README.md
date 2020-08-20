## Parallel-Programming

This repository contains codes for the assignments of _High Performance Parallel Programming (CS61064)_ at IIT Kharagpur during Spring 2020. Here is a brief overview
of what will be found inside the folders.


| Name | Brief description |
| :--: | :-- |
| CUDA | NVIDIA GPU kernel implementations (CUDA C) for different compute operations like **Reduction**, **2D Convolution**, **Matrix Transpose** and **Dot Product**. Exploits different concepts like Thread packing in blocks, Global memory access coalescing, Shared memory accesses and bank conflicts to reduce overhead in (typically) *Tesla K40* or *K80* GPUs. |
| OpenMP | Implementation (C) for performing rotation of an object (given in terms of points) about a given axis in 3D cartesian coordinates using **parallelized matrix multiplication** operations. |
| MPI | Distributed memory (MPI C) implementation for **Histogram equalisation** and **Sobel Filtering** of an input image. |


The Colab Notebooks folder contains some experiments I performed to ensure the proper functioning of the kernels and the correctness of the shared memory
optimisations. (basically _debugging!_) These notebooks contain [2D Convolution (naive and shared memory implementations)](https://github.com/swag2198/Parallel-Programming/blob/master/Colab%20Notebooks/conv2D.ipynb), [Matrix Transpose](https://github.com/swag2198/Parallel-Programming/blob/master/Colab%20Notebooks/transpose.ipynb) and [Dot Product Reduction](https://github.com/swag2198/Parallel-Programming/blob/master/Colab%20Notebooks/dotproduct.ipynb)
kernels. The notebooks also contain detailed `nvprof` profiling and `CUDA MEMCHECK` checks for the GPU codes.

It is worthwhile to note that much of the optimisations involving global memory access coalescing did not work as expected for later generation GPUs like
_Pascal_, _Maxwell_ and _Turing_. I found this stack overflow [post](https://stackoverflow.com/questions/56142674/memory-coalescing-and-nvprof-results-on-nvidia-pascal) that also tries to explain the anomaly that I encountered.
Another useful SO post that explains nvprof option for bandwidth is [here](https://stackoverflow.com/questions/37732735/nvprof-option-for-bandwidth).

