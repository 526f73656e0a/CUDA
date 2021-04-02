
/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : assignment 2
///
///  project    : GPU Programming
///
///  description: CUDA basics
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include<algorithm>
#include <limits>
#include <cassert>
#include <iomanip>
#include<numeric>
#include "cuda_util.h"
#include<cuda_runtime.h>
#include<cuda.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <chrono>

typedef std::chrono::nanoseconds TimeT;
////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction for sum of array, global memory version
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
reduction1(int*data,const int size, const int stride){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid * stride] += data[(tid * stride) + (stride / 2)];

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction for sum of array, shared memory version /inefficient, % is costly/
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
reduction2(int* data, const int size) {
    extern __shared__ int share[];

    share[threadIdx.x]=data[blockIdx.x*blockDim.x+threadIdx.x];
    __syncthreads();

    for(int s = 1; s<blockDim.x; s*=2){
        if(threadIdx.x % (2*s)==0){
            share[threadIdx.x]+=share[threadIdx.x+s];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        data[blockIdx.x]=share[0];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction for sum of array, better shared memory version
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
reduction2_1(int* data, const int size) {
    extern __shared__ int share[];

    share[threadIdx.x]=data[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    for(int i = 1;i<blockDim.x;i*=2){
        int idx = 2*i*threadIdx.x;

        if(idx<blockDim.x){
            share[idx]+=share[idx+i];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        data[blockIdx.x]=share[0];
    }
    // TODO: implement reduction

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction for sum of array, version avoiding bank conflicts and div branches
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
reduction3(int* data, const int size) {
    extern __shared__ int share[];

    share[threadIdx.x]=data[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    for(int i = blockDim.x /2; i>0; i>>=1){
        //int idx = 2*i*threadIdx.x;

        if(threadIdx.x<i){
            share[threadIdx.x]+=share[threadIdx.x+i];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        data[blockIdx.x]=share[0];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction for sum of array, version avoiding bank conflicts /Works with half the thread blocks/
// !!!!!!!!!!!!!!!! If used divide the thread blocks in version3 by 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
reduction3_1(int* data, const int size) {
    __shared__ int share[1024*4];

    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int ind = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

// Store first partial result instead of just the elements
    share[threadIdx.x] = data[ind] + data[ind + blockDim.x];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < i) {
            share[threadIdx.x] += share[threadIdx.x + i];
        }
        __syncthreads();
}

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        data[blockIdx.x] = share[0];
    }


    }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Unrolling part of loop
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__
void
warpReduce_0(volatile int* share, int t) {
	share[t] += share[t + 8];
	share[t] += share[t + 4];
	share[t] += share[t + 2];
	share[t] += share[t + 1];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction for sum of array, last warps unrolled
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
reduction4_0(int* data, const int size) {
    // Allocate shared memory
    extern __shared__ int share[];

    share[threadIdx.x]=data[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    for(int i = blockDim.x /2; i>8; i>>=1){
        //int idx = 2*i*threadIdx.x;

        if(threadIdx.x<i){
            share[threadIdx.x]+=share[threadIdx.x+i];
        }
        __syncthreads();
    }

    if(threadIdx.x<8){
        warpReduce_0(share,threadIdx.x);
    }

    if(threadIdx.x==0){
        data[blockIdx.x]=share[0];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// last loops unrolled
////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned int block>
__device__
void
warpReduce(volatile int* share, int tid){
    if(block>=64) share[tid]+=share[tid+32];
    if(block>=32) share[tid]+=share[tid+16];
    if(block>=16) share[tid]+=share[tid+8];
    if(block>=8) share[tid]+=share[tid+4];
    if(block>=4) share[tid]+=share[tid+2];
    if(block>=2) share[tid]+=share[tid+1];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction for sum of array, all warps unrolled
////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned int block>
__global__
void
reduction4(int* data, const int size) {
    extern __shared__ int share[];
    int id = blockIdx.x*(blockDim.x*2)+threadIdx.x;
    share[threadIdx.x]=data[id]+data[id+blockDim.x];
    __syncthreads();
    if(block>=1024){
    if(threadIdx.x<512){share[threadIdx.x]+=share[threadIdx.x+512];__syncthreads();}
    }
    if(block>=512){
    if(threadIdx.x<256){share[threadIdx.x]+=share[threadIdx.x+256];__syncthreads();}
    }
    if(block>=256){
    if(threadIdx.x<128){share[threadIdx.x]+=share[threadIdx.x+128];__syncthreads();}
    }
    if(block>=128){
    if(threadIdx.x<64){share[threadIdx.x]+=share[threadIdx.x+64];__syncthreads();}
    }
    if(threadIdx.x<32) warpReduce<block>(share,threadIdx.x);

    __syncthreads();
    if(threadIdx.x==0){
        data[blockIdx.x]=share[0];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute reference solution for reduction
// @return sum of array
////////////////////////////////////////////////////////////////////////////////////////////////////
int
reference(int* data, const int size) {
    int ref = std::accumulate(data,data+size,0);
    return ref;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Generate input for reduction
////////////////////////////////////////////////////////////////////////////////////////////////////
void
input(int*& data, const int size) {

    data = (int*)malloc(sizeof(int) * size);
    //printf("hi");
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize Cuda device
////////////////////////////////////////////////////////////////////////////////////////////////////
void
initDevice(int& device_handle, int& max_threads_per_block) {

    // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize device memory
////////////////////////////////////////////////////////////////////////////////////////////////////
void
initDeviceMemory(int* data, int*& data_device, const int size) {

    // TODO
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//! Global memory version
////////////////////////////////////////////////////////////////////////////////////////////////////
void
version1(const int size) {
    std::cout<<"Beginning execution for version1"<<std::endl;

    int* data = nullptr;
    int* data_ref = nullptr;
    int ref = 0;
    bool Corr = true;
    input(data, size);
    input(data_ref, size);
    memcpy(data_ref,data,size);
    // TODO: compute reference solution
    ref = reference(data_ref,size);
    //std::cout << "ref = " << ref << std::endl;
    // TODO: query available and set device

    int deviceCount = 0;
    checkErrorsCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) return;
    //----
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    //printDeviceProps(devProp);
    //----
    int device_handle = 0;
    cudaSetDevice(device_handle);

    // TODO: initialize device memory
    int* data_device = nullptr;
    //---
    checkErrorsCuda(cudaMalloc((void**)&data_device, sizeof(int) * size));
    //---
    checkErrorsCuda(cudaMemcpy((void*)data_device, data, sizeof(int) * size, cudaMemcpyHostToDevice));

    // TODO: determine thread layout
    const unsigned int MAX_THREADS_PER_BLOCK = 1024;

    int num_threads_per_block = std::min((const unsigned int)size, MAX_THREADS_PER_BLOCK);
    int num_blocks = size / (MAX_THREADS_PER_BLOCK*2);
    if (size % MAX_THREADS_PER_BLOCK != 0) {
        num_blocks++;
    }
    //std::cout << "num_blocks = " << num_blocks << " :: "
    //<< "num_threads_per_block = " << num_threads_per_block << std::endl;
    int stride = 2;
    // TODO: run kernel
    auto start = std::chrono::steady_clock::now();
    for (int i = 0;i<log2(size/2)+1;i++){
        reduction1 <<<num_blocks, num_threads_per_block >>> (data_device,size,stride);
        stride = stride*2;
        if(num_blocks!=1){
            num_blocks=num_blocks/2;
            //std::cout<<"num_blocks change to: "<<num_blocks<<std::endl;
        }
        else if(num_blocks==1&&num_threads_per_block>=2){
            num_threads_per_block=num_threads_per_block/2;
            //std::cout<<"num_threads_per_block change to: "<<num_threads_per_block<<std::endl;
        }
        //std::cout<<i<<std::endl;
    }
    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();



    // TODO: copy result back to host and check correctness
    checkErrorsCuda(cudaMemcpy(data, data_device, sizeof(int) * size, cudaMemcpyDeviceToHost));

    /*
    std::cout<<"The value of data[0]: "<<data[0]<<std::endl
    <<"The value of data[1]: "<<data[1]<<std::endl
    <<"The value of ref: "<<ref<<std::endl;
    */

    //std::cout<<time<< std::endl;

    ///*
    std::cout <<  "\nThe execution of version 1 is over\n"
    << "The code execution" << ((data[0] == ref) ? " was successfull" : " has failed")
    << "\nIt took " << time << "nanoseconds" << std::endl;
    //*/
    // TODO: clean up device memory
    checkErrorsCuda(cudaFree(data_device));
    // clean up host memory
    free(data);
    free(data_ref);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Shared memory version
////////////////////////////////////////////////////////////////////////////////////////////////////
void
version2(const int size) {
    std::cout<<"\n\nBeginning execution for version2"<<std::endl;

    int* data = nullptr;
    int* data_ref = nullptr;
    input(data, size);
    input(data_ref,size);
    memcpy(data_ref,data,size);
    // TODO: compute reference solution
    int ref = 0;
    ref = reference(data_ref,size);

    // TODO: query available and set device

    int deviceCount = 0;
    checkErrorsCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) return;
    //----
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    //printDeviceProps(devProp);
    //----
    int device_handle = 0;
    cudaSetDevice(device_handle);

    // TODO: initialize device memory

    int* data_device = nullptr;
    //---
    checkErrorsCuda(cudaMalloc((void**)&data_device, sizeof(int) * size));
    //---
    checkErrorsCuda(cudaMemcpy((void*)data_device, data, sizeof(int) * size, cudaMemcpyHostToDevice));

    // TODO: determine thread layout

    const unsigned int MAX_THREADS_PER_BLOCK = 1024;

    int num_threads_per_block = std::min((const unsigned int)size, MAX_THREADS_PER_BLOCK);
    int num_blocks = size / (MAX_THREADS_PER_BLOCK);
    if (size % MAX_THREADS_PER_BLOCK != 0) {
        num_blocks++;
    }
    //std::cout << "num_blocks = " << num_blocks << " :: "
    //<< "num_threads_per_block = " << num_threads_per_block << std::endl;

    auto start = std::chrono::steady_clock::now();
    // TODO: run kernel

    reduction2<<<num_blocks,num_threads_per_block,num_threads_per_block*sizeof(int)>>>(data_device,size);
    //each block holds a sum for its threads, now just sum the blocks
    reduction2<<<1,num_blocks,num_blocks*sizeof(int)>>>(data_device,size);

    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();

    checkErrorsCuda(cudaMemcpy(data, data_device, sizeof(int) * size, cudaMemcpyDeviceToHost));
    /*
    std::cout<<"The value of data[0]: "<<data[0]<<std::endl
    <<"The value of data[1]: "<<data[1]<<std::endl
    <<"The value of ref: "<<ref<<std::endl;
    */
    // TODO: copy result back to host and check correctness

    //std::cout<< time<< std::endl;

    ///*
    std::cout <<  "\nThe execution of version 2 is over\n"
    << "The code execution" << ((data[0] == ref) ? " was successfull" : " has failed")
    << "\nIt took " << time << "nanoseconds" << std::endl;
    //*/

    // TODO: clean up device memory
    checkErrorsCuda(cudaFree(data_device));

    // clean up host memory
    free(data);
    free(data_ref);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! version avoiding divegent branching
////////////////////////////////////////////////////////////////////////////////////////////////////
void
version3(const int size) {
    std::cout<<"\n\nBeginning execution for version3"<<std::endl;

    int* data = nullptr;
    int* data_ref = nullptr;
    input(data, size);
    input(data_ref,size);
    memcpy(data_ref,data,size);
    // TODO: compute reference solution
    int ref = 0;
    ref = reference(data_ref,size);

    // TODO: query available and set device

    int deviceCount = 0;
    checkErrorsCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) return;
    //----
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    //printDeviceProps(devProp);
    //----
    int device_handle = 0;
    cudaSetDevice(device_handle);

    // TODO: initialize device memory

    int* data_device = nullptr;
    //---
    checkErrorsCuda(cudaMalloc((void**)&data_device, sizeof(int) * size));
    //---
    checkErrorsCuda(cudaMemcpy((void*)data_device, data, sizeof(int) * size, cudaMemcpyHostToDevice));

    // TODO: determine thread layout

    const unsigned int MAX_THREADS_PER_BLOCK = 1024;

    int num_threads_per_block = std::min((const unsigned int)size, MAX_THREADS_PER_BLOCK);
    int num_blocks = size / (MAX_THREADS_PER_BLOCK);
    if (size % MAX_THREADS_PER_BLOCK != 0) {
        num_blocks++;
    }
    //std::cout << "num_blocks = " << num_blocks << " :: "
    //<< "num_threads_per_block = " << num_threads_per_block << std::endl;

    auto start = std::chrono::steady_clock::now();
    // TODO: run kernel
    //divide num_blocks by 2 with reduction3_1
    reduction3<<<num_blocks,num_threads_per_block, num_threads_per_block*sizeof(int)>>>(data_device,size);
    //each block holds a sum for its threads, now just sum the blocks
    reduction3<<<1,num_blocks,num_threads_per_block*sizeof(int)>>>(data_device,size);

    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();

    checkErrorsCuda(cudaMemcpy(data, data_device, sizeof(int) * size, cudaMemcpyDeviceToHost));
    /*
    std::cout<<"The value of data[0]: "<<data[0]<<std::endl
    <<"The value of data[1]: "<<data[1]<<std::endl
    <<"The value of ref: "<<ref<<std::endl;
    */
    // TODO: copy result back to host and check correctness

    //std::cout<< time<< std::endl;

    ///*
    std::cout <<  "\nThe execution of version 3 is over\n"
    << "The code execution" << ((data[0] == ref) ? " was successfull" : " has failed")
    << "\nIt took " << time << "nanoseconds" << std::endl;
    //*/


    // TODO: clean up device memory
    checkErrorsCuda(cudaFree(data_device));

    // clean up host memory
    free(data);
    free(data_ref);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//! version with unrolled loops
////////////////////////////////////////////////////////////////////////////////////////////////////
void
version4(const int size) {
    std::cout<<"\n\nBeginning execution for version4"<<std::endl;

    int* data = nullptr;
    int* data_ref = nullptr;
    input(data, size);
    input(data_ref,size);
    memcpy(data_ref,data,size);
    // TODO: compute reference solution
    int ref = 0;
    ref = reference(data_ref,size);

    // TODO: query available and set device

    int deviceCount = 0;
    checkErrorsCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) return;
    //----
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    //printDeviceProps(devProp);
    //----
    int device_handle = 0;
    cudaSetDevice(device_handle);

    // TODO: initialize device memory

    int* data_device = nullptr;
    //---
    checkErrorsCuda(cudaMalloc((void**)&data_device, sizeof(int) * size));
    //---
    checkErrorsCuda(cudaMemcpy((void*)data_device, data, sizeof(int) * size, cudaMemcpyHostToDevice));

    // TODO: determine thread layout

    const unsigned int MAX_THREADS_PER_BLOCK = 1024;

    int num_threads_per_block = std::min((const unsigned int)size, MAX_THREADS_PER_BLOCK);
    int num_blocks = size / (MAX_THREADS_PER_BLOCK);
    if (size % MAX_THREADS_PER_BLOCK != 0) {
        num_blocks++;
    }
    //std::cout << "num_blocks = " << num_blocks << " :: "
    //<< "num_threads_per_block = " << num_threads_per_block << std::endl;

    auto start = std::chrono::steady_clock::now();
    // TODO: run kernel
    //--- not all unwrapped
    reduction4_0<<<num_blocks,num_threads_per_block,num_threads_per_block*sizeof(int)>>>(data_device,size);
    //each block holds a sum for its threads, now just sum the blocks
    reduction4_0<<<1,num_blocks,num_threads_per_block*sizeof(int)>>>(data_device,size);
    //---

    // All loops unwrapped
    /*
    dim3 dimBlock(num_threads_per_block,1,1);
    dim3 dimGrid(num_blocks,1,1);
    int smemSize = num_threads_per_block  *sizeof(int);
    switch(num_threads_per_block){
        case 1024:
        reduction4<1024><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 512:
        reduction4<512><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 256:
        reduction4<256><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 128:
        reduction4<128><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 64:
        reduction4<64><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 32:
        reduction4<32><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 16:
        reduction4<16><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 8:
        reduction4<8><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 4:
        reduction4<4><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 2:
        reduction4<2><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
        case 1:
        reduction4<1><<<dimGrid,dimBlock,smemSize>>>(data_device,size);break;
    }
    */
    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();

    checkErrorsCuda(cudaMemcpy(data, data_device, sizeof(int) * size, cudaMemcpyDeviceToHost));
    /*
    std::cout<<"The value of data[0]: "<<data[0]<<std::endl
    <<"The value of data[1]: "<<data[1]<<std::endl
    <<"The value of ref: "<<ref<<std::endl;
    */
    // TODO: copy result back to host and check correctness

    //std::cout<< time<< std::endl;

    ///*
    std::cout <<  "\nThe execution of version 4 is over\n"
    << "The code execution" << ((data[0] == ref) ? " was successfull" : " has failed")
    << "\nIt took " << time << " nanoseconds" << std::endl;
    //*/


    // TODO: clean up device memory
    checkErrorsCuda(cudaFree(data_device));

    // clean up host memory
    free(data);
    free(data_ref);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// program entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main(int /*argc*/, char** /*argv*/) {
    const int n = 16384; //16384,32768,65536,131072,262144,524288,1048576
    std::cout<<"Num of elements :: " <<n<<std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Output is in nanoseconds for better comparison
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    version1(n);
    version2(n);
    version3(n);
    version4(n);
    return EXIT_SUCCESS;
}
