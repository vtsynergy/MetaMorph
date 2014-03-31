#include <stdio.h>

#include "afosr_cfd_cuda_core.cuh"


__device__ void block_reduction(double *psum, int tid, int len_) {
	int stride = len_ >> 1; 
	while (stride > 0) {
	//while (stride > 32) {
		if (tid  < stride) psum[tid] += psum[tid+stride];
		__syncthreads(); 
		stride >>= 1;
	}
	__syncthreads();      
	/*if (tid < 32) { 
		psum[tid] += psum[tid+32];
		__syncthreads();
		psum[tid] += psum[tid+16];
		__syncthreads();
		psum[tid] += psum[tid+8];
		__syncthreads();
		psum[tid] += psum[tid+4];
		__syncthreads();
		psum[tid] += psum[tid+2];
		__syncthreads();
		psum[tid] += psum[tid+1];
		__syncthreads();
	}*/
}


//Paul - Implementation of double atomicAdd from CUDA Programming Guide: Appendix B.12
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

// this kernel works for 3D data only.
//  PHI1 and PHI2 are input arrays.
//  s* parameters are start values in each dimension.
//  e* parameters are end values in each dimension.
//  s* and e* are only necessary when the halo layers 
//    has different thickness along various directions.
//  i,j,k are the array dimensions
//  len_ is number of threads in a threadblock.
//       This can be computed in the kernel itself.
__global__ void kernel_dotProd(double *phi1, double *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez, 
		int gz, double * reduction, int len_) {
	extern __shared__ double psum[];
	int tid, loads, x, y, z, itr;
	bool boundx, boundy, boundz;
	tid = threadIdx.x+(threadIdx.y)*blockDim.x+(threadIdx.z)*(blockDim.x*blockDim.y);

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));

	for (itr = 0; itr < loads; itr++) {
		z = itr*blockDim.z+threadIdx.z +sz;
		boundz = ((z >= sz) && (z <= ez));
		//if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j];
		if (boundx && boundy && boundz) psum[tid] += phi1[x*i+y+z*i*j] * phi2[x*i+y+z*i*j];
	}

	__syncthreads();
	//After accumulating the Z-dimension, have each block internally reduce X and Y
	block_reduction(psum,tid,len_);
	__syncthreads();

	//Merge reduced values from all blocks
	if(tid == 0) atomicAdd(reduction,psum[0]);
}


__global__ void kernel_reduction3(double *phi,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez, 
		int gz, double * reduction, int len_) {
	extern __shared__ double psum[];
	int tid, loads, x, y, z, itr;
	bool boundx, boundy, boundz;
	tid = threadIdx.x+(threadIdx.y)*blockDim.x+(threadIdx.z)*(blockDim.x*blockDim.y);

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));

	for (itr = 0; itr < loads; itr++) {
		z = itr*blockDim.z+threadIdx.z +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi[x*i+y+z*i*j];
	}

	__syncthreads();
	//After accumulating the Z-dimension, have each block internally reduce X and Y
	block_reduction(psum,tid,len_);
	__syncthreads();

	//Merge reduced values from all blocks
	if(tid == 0) atomicAdd(reduction,psum[0]);
}


cudaError_t cuda_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], double * data1, double * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val, int async, cudaEvent_t ((*event)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_size = sizeof(double) * (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
	//Lingering debugging output
	//printf("Grid: %d %d %d\n", grid.x, grid.y, grid.z);
	//printf("Block: %d %d %d\n", block.x, block.y, block.z);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_size);
	if (event != NULL) {
		cudaEventCreate(&(*event)[0]);
		cudaEventRecord((*event)[0], 0);
	}
	kernel_dotProd<<<grid,block,smem_size>>>(data1, data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[1], (*arr_end)[0], (*arr_end)[2], (*grid_size)[2], reduced_val, (*block_size)[0] * (*block_size)[1] * (*block_size)[2]);
	if (event != NULL) {
		cudaEventCreate(&(*event)[1]);
		cudaEventRecord((*event)[1], 0);
	}
	if (!async) ret = cudaThreadSynchronize();
	return(ret);
}


cudaError_t cuda_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], double * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val, int async, cudaEvent_t ((*event)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_size = sizeof(double) * (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
	//Lingering debugging output
	//printf("Grid: %d %d %d\n", grid.x, grid.y, grid.z);
	//printf("Block: %d %d %d\n", block.x, block.y, block.z);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_size);
	if (event != NULL) {
		cudaEventCreate(&(*event)[0]);
		cudaEventRecord((*event)[0], 0);
	}
	kernel_reduction3<<<grid,block,smem_size>>>(data, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[1], (*arr_end)[0], (*arr_end)[2], (*grid_size)[2], reduced_val, (*block_size)[0] * (*block_size)[1] * (*block_size)[2]);
	if (event != NULL) {
		cudaEventCreate(&(*event)[1]);
		cudaEventRecord((*event)[1], 0);
	}
	if (!async) ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return(ret);
}
