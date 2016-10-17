#include <stdio.h>

/** CUDA Back-End **/
#include "mm_cuda_backend.cuh"

// non-specialized class template
template<typename T>
class SharedMem {
public:
	// Ensure that we won't compile any un-specialized types
	__device__
	T* getPointer() {
		return (T*) NULL;
	}
	;
};

// specialization for double
template<>
class SharedMem<double> {
public:
	__device__
	double* getPointer() {
		extern __shared__ double s_db[]; return s_db;
	}
};

// specialization for float
template<>
class SharedMem<float> {
public:
	__device__
	float* getPointer() {
		extern __shared__ float s_fl[]; return s_fl;
	}
};

// specialization for unsigned long long
template<>
class SharedMem<unsigned long long> {
public:
	__device__
	unsigned long long* getPointer() {
		extern __shared__ unsigned long long s_ul[]; return s_ul;
	}
};

// specialization for int
template<>
class SharedMem<int> {
public:
	__device__
	int* getPointer() {
		extern __shared__ int s_in[]; return s_in;
	}
};

// specialization for unsigned int
template<>
class SharedMem<unsigned int> {
public:
	__device__
	unsigned int* getPointer() {
		extern __shared__ unsigned int s_ui[]; return s_ui;
	}
};

template <typename T>
__device__ void block_reduction(T *psum, int tid, int len_) {
	int stride = len_ >> 1;
	while (stride > 0) {
		//while (stride > 32) {
		if (tid < stride) psum[tid] += psum[tid+stride];
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

//TODO figure out how to use templates with the __X_as_Y intrinsics
//Paul - Implementation of double atomicAdd from CUDA Programming Guide: Appendix B.12
__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

//wrapper for all the types natively supported by CUDA
template <typename T> __device__ T atomicAdd_wrapper(T* addr, T val) {
	return atomicAdd(addr, val);
}

//wrapper for had-implemented double type
template <> __device__ double atomicAdd_wrapper<double>(double* addr, double val) {
	return atomicAdd(addr, val);
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
template <typename T>
__global__ void kernel_dotProd(T *phi1, T *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, T * reduction, int len_) {
	SharedMem<T> shared;
	T * psum = shared.getPointer();
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
		if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j];
	}

	__syncthreads();
	//After accumulating the Z-dimension, have each block internally reduce X and Y
	block_reduction<T>(psum,tid,len_);
	__syncthreads();

	//Merge reduced values from all blocks
	if(tid == 0) atomicAdd_wrapper<T>(reduction,psum[0]);
}

template <typename T>
__global__ void kernel_reduction3(T *phi,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, T * reduction, int len_) {
	SharedMem<T> shared;
	T *psum = shared.getPointer();
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
		if (boundx && boundy && boundz) psum[tid] += phi[x+y*i+z*i*j];
	}

	__syncthreads();
	//After accumulating the Z-dimension, have each block internally reduce X and Y
	block_reduction<T>(psum,tid,len_);
	__syncthreads();

	//Merge reduced values from all blocks
	if(tid == 0) atomicAdd_wrapper<T>(reduction,psum[0]);
}

//"constant" buffers for face indexing in pack/unpack kernels
__constant__ int c_face_size[METAMORPH_FACE_MAX_DEPTH];
__constant__ int c_face_stride[METAMORPH_FACE_MAX_DEPTH];
//Size of all children (>= level+1) so at level 0, child_size = total_num_face_elements
__constant__ int c_face_child_size[METAMORPH_FACE_MAX_DEPTH];

//Helper function to compute the integer read offset for buffer packing
//TODO: Add support for multi-dimensional grid/block
__device__ int get_pack_index(int tid, int * a, int start, int count) {
	int pos;
	int i, j, k, l;
	for (i = 0; i < count; i++)
		a[tid % blockDim.x + i * blockDim.x] = 0;

	for (i = 0; i < count; i++) {
		k = 0;
		for (j = 0; j < i; j++) {
			k += a[tid % blockDim.x + j * blockDim.x] * c_face_child_size[j];
		}
		l = c_face_child_size[i];
		for (j = 0; j < c_face_size[i]; j++) {
			if (tid - k < l)
				break;
			else
				l += c_face_child_size[i];
		}
		a[tid % blockDim.x + i * blockDim.x] = j;
	}
	pos = start;
	for (i = 0; i < count; i++) {
		pos += a[tid % blockDim.x + i * blockDim.x] * c_face_stride[i];
	}
	return pos;
}

template <typename T>
__global__ void kernel_pack(T *packed_buf, T *buf, int size, int start, int count)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int nthreads = gridDim.x * blockDim.x;
	extern __shared__ int a[];
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count)];
}

template <typename T>
__global__ void kernel_unpack(T *packed_buf, T *buf, int size, int start, int count)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int nthreads = gridDim.x * blockDim.x;
	extern __shared__ int a[];
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	buf[get_pack_index(idx, a, start, count)] = packed_buf[idx];
}

//TODO: Expand to multiple transpose elements per thread
//#define TRANSPOSE_TILE_DIM (16)
//#define TRANSPOSE_BLOCK_ROWS (16)
template <typename T>
__global__ void kernel_transpose_2d(T *odata, T *idata, int arr_width, int arr_height, int tran_width, int tran_height)
{
	SharedMem<T> shared;
	T * tile = shared.getPointer();
	//__shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];
	//__shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM];

	int blockIdx_x, blockIdx_y;
	int gridDim_x, gridDim_y;

// do diagonal reordering
	//The if case degenerates to the else case, no need to have both
	//if (width == height)
	//{
	//    blockIdx_y = blockIdx.x;
	//    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
	//}
	//else
	//{
	//First figure out your number among the actual grid blocks
	int bid = blockIdx.x + gridDim.x*blockIdx.y;
	//Then figure out how many logical blocks are required in each dimension
	gridDim_x = (tran_width-1+blockDim.x)/blockDim.x;
	gridDim_y = (tran_height-1+blockDim.y)/blockDim.y;
	//Then how many logical and actual grid blocks
	int logicalBlocks = gridDim_x*gridDim_y;
	int gridBlocks = gridDim.x*gridDim.y;
	//Loop over all logical blocks
	for (; bid < logicalBlocks; bid += gridBlocks) {
		//Compute the current logical block index in each dimension
		blockIdx_y = bid%gridDim_y;
		blockIdx_x = ((bid/gridDim_y)+blockIdx_y)%gridDim_x;
		//}

		//int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + threadIdx.x;
		int xIndex_in = blockIdx_x * blockDim.x + threadIdx.x;
		//int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + threadIdx.y;
		int yIndex_in = blockIdx_y * blockDim.y + threadIdx.y;
		//int index_in = xIndex_in + (yIndex_in)*width;
		int index_in = xIndex_in + (yIndex_in)*arr_width;

		//int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + threadIdx.x;
		int xIndex_out = blockIdx_y * blockDim.y + threadIdx.x;
		//int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + threadIdx.y;
		int yIndex_out = blockIdx_x * blockDim.x + threadIdx.y;
		//int index_out = xIndex_out + (yIndex_out)*height;
		int index_out = xIndex_out + (yIndex_out)*arr_height;

		//The blockDim.x+1 in R/W steps is to avoid bank conflicts if blockDim.x==16 or 32
		if(xIndex_in < tran_width && yIndex_in < tran_height)
		//tile[threadIdx.y+1][threadIdx.x] = idata[index_in];
		tile[threadIdx.x+(blockDim.x+1)*threadIdx.y] = idata[index_in];
		//tile[threadIdx.y][threadIdx.x] = idata[index_in];

		__syncthreads();

		if(xIndex_out < tran_height && yIndex_out < tran_width)
		//if(xIndex_out < tran_width && yIndex_out < tran_height)
		//odata[index_out] = tile[threadIdx.x][threadIdx.y];
		odata[index_out] = tile[threadIdx.y+(blockDim.y+1)*threadIdx.x];
		//odata[index_out] = tile[threadIdx.x][threadIdx.y];

		//Added when the loop was added to ensure writes are finished before new vals go into shared memory
		__syncthreads();
	}
}

// this kernel works for 3D data only.
//  i,j,k are the array dimensions
//  s* parameters are start values in each dimension.
//  e* parameters are end values in each dimension.
//  s* and e* are only necessary when the halo layers
//    has different thickness along various directions.
//  len_ is number of threads in a threadblock.
//       This can be computed in the kernel itself.
template <typename T>

//Read-only cache + Rigster blocking (Z) + smem blocking (X-Y)
// work only with 2D thread blocks (use rectangular blocks, i.e. 64*4, 128*2)
__global__ void kernel_stencil_3d7p(const T * __restrict__ ind, T * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_) {
	SharedMem<T> shared;
	T * bind = shared.getPointer(); //[blockDim.x+2*blockDim.y+2]
	const int bi = (blockDim.x+2);
	const int bc = (threadIdx.x+1)+(threadIdx.y+1)*bi;
	T r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;
	z = threadIdx.z +sz;//blockDim.z ==1
	c = x+y*i+z*ij;
	r0 = ind[c];
	rz1 = ind[c-ij];
	rz2 = ind[c+ij];

	boundy = ((y > sy) && (y < ey));
	boundx = ((x > sx) && (x < ex));
#pragma unroll 8
	for (; z < gz; z++) {
		boundz = ((z > sz) && (z < ez));
		bind[bc] = r0;

		if(threadIdx.x == 0)
		bind[bc-1] = ind[c-1];
		else if (threadIdx.x == blockDim.x-1)
		bind[bc+1] = ind[c+1];

		if(threadIdx.y == 0)
		bind[bc-bi] = ind[c-i];
		else if (threadIdx.y == blockDim.y-1)
		bind[bc+bi] = ind[c+i];

		__syncthreads();

		if (boundx && boundy && boundz)
		outd[c] = ( rz1 + bind[bc-1] + bind[bc-bi] + r0 +
				bind[bc+bi] + bind[bc+1] + rz2 ) / (T) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];

		__syncthreads();
	}
}

#if 0
template <typename T>
// work with 2D and 3D thread blocks
__global__ void kernel_stencil_3d7p_v0(T *ind, T *outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_) {
	int x, y, z, itr;
	bool boundx, boundy, boundz;

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;

	boundy = ((y > sy) && (y < ey));
	boundx = ((x > sx) && (x < ex));

	for (itr = 0; itr < gz; itr++) {
		z = itr*blockDim.z+threadIdx.z +sz;
		boundz = ((z > sz) && (z < ez));
		if (boundx && boundy && boundz)
		outd[x+y*i+z*i*j] = ( ind[x+y*i+(z-1)*i*j] + ind[(x-1)+y*i+z*i*j] + ind[x+(y-1)*i+z*i*j] +
				ind[x+y*i+z*i*j] + ind[x+(y+1)*i+z*i*j] + ind[(x+1)+y*i+z*i*j] +
				ind[x+y*i+(z+1)*i*j] ) / (T) 7;
	}
}

template <typename T>
//Read-only cache + Rigster blocking (Z)
// work only with 2D thread blocks
__global__ void kernel_stencil_3d7p_v1(const T * __restrict__ ind, T * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_) {
	T r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;
	z = threadIdx.z +sz; //blockDim.z ==1
	c = x+y*i+z*ij;
	r0 = ind[c];
	rz1 = ind[c-ij];
	rz2 = ind[c+ij];

	boundy = ((y > sy) && (y < ey));
	boundx = ((x > sx) && (x < ex));
#pragma unroll 8
	for (; z < gz; z++) {
		boundz = ((z > sz) && (z < ez));
		if (boundx && boundy && boundz)
		outd[c] = ( rz1 + ind[c-1] + ind[c-i] + r0 +
				ind[c+i] + ind[c+1] + rz2 ) / (T) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
	}
}

template <typename T>
//Read-only cache + Rigster blocking (Z) + manual prefetch
// work only with 2D thread blocks
__global__ void kernel_stencil_3d7p_v2(const T * __restrict__ ind, T * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_) {
	T r0, rz1, rz2, rz3;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;
	z = threadIdx.z +sz; //blockDim.z ==1
	c = x+y*i+z*ij;
	r0 = ind[c];
	rz1 = ind[c-ij];
	rz2 = ind[c+ij];
	rz3 = ind[c+ij*2];

	boundy = ((y > sy) && (y < ey));
	boundx = ((x > sx) && (x < ex));
#pragma unroll 8
	for (; z < gz; z++) {
		boundz = ((z > sz) && (z < ez));
		if (boundx && boundy && boundz)
		outd[c] = ( rz1 + ind[c-1] + ind[c-i] + r0 +
				ind[c+i] + ind[c+1] + rz2 ) / (T) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = rz3;
		rz3 = ind[c+ij*2];
	}
}

template <typename T>

//Read-only cache + Rigster blocking (Z) + + smem blocking (X-Y)
// work only with 2D thread blocks
__global__ void kernel_stencil_3d7p_v3(const T * __restrict__ ind, T * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_) {
	SharedMem<T> shared;
	T * bind = shared.getPointer(); //[blockDim.x+2*blockDim.y+2]
	const int bi = (blockDim.x+2);
	const int bc = (threadIdx.x+1)+(threadIdx.y+1)*bi;
	T r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;
	z = threadIdx.z +sz;//blockDim.z ==1
	c = x+y*i+z*ij;
	r0 = ind[c];
	rz1 = ind[c-ij];
	rz2 = ind[c+ij];

	boundy = ((y > sy) && (y < ey));
	boundx = ((x > sx) && (x < ex));
#pragma unroll 8
	for (; z < gz; z++) {
		boundz = ((z > sz) && (z < ez));
		bind[bc] = r0;

		if(threadIdx.x == 0)
		bind[bc-1] = ind[c-1];
		else if (threadIdx.x == blockDim.x-1)
		bind[bc+1] = ind[c+1];

		if(threadIdx.y == 0)
		bind[bc-bi] = ind[c-i];
		else if (threadIdx.y == blockDim.y-1)
		bind[bc+bi] = ind[c+i];

		__syncthreads();

		if (boundx && boundy && boundz)
		outd[c] = ( rz1 + bind[bc-1] + bind[bc-bi] + r0 +
				bind[bc+bi] + bind[bc+1] + rz2 ) / (T) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
	}
}

template <typename T>

// explicit Read-only cache + Rigster blocking (Z) + + smem blocking (X-Y)
// work only with 2D thread blocks
__global__ void kernel_stencil_3d7p_v4(const T * __restrict__ ind, T * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_) {
	SharedMem<T> shared;
	T * bind = shared.getPointer(); //[blockDim.x+2*blockDim.y+2]
	const int bi = (blockDim.x+2);
	const int bc = (threadIdx.x+1)+(threadIdx.y+1)*bi;
	T r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (blockIdx.x)*blockDim.x+threadIdx.x+sx;
	y = (blockIdx.y)*blockDim.y+threadIdx.y+sy;
	z = threadIdx.z +sz;//blockDim.z ==1
	c = x+y*i+z*ij;
	r0 = __ldg(&ind[c]);
	rz1 = __ldg(&ind[c-ij]);
	rz2 = __ldg(&ind[c+ij]);

	boundy = ((y > sy) && (y < ey));
	boundx = ((x > sx) && (x < ex));
#pragma unroll 8
	for (; z < gz; z++) {
		boundz = ((z > sz) && (z < ez));
		bind[bc] = r0;

		if(threadIdx.x == 0)
		bind[bc-1] = __ldg(&ind[c-1]);
		else if (threadIdx.x == blockDim.x-1)
		bind[bc+1] = __ldg(&ind[c+1]);

		if(threadIdx.y == 0)
		bind[bc-bi] = __ldg(&ind[c-i]);
		else if (threadIdx.y == blockDim.y-1)
		bind[bc+bi] = __ldg(&ind[c+i]);

		__syncthreads();

		if (boundx && boundy && boundz)
		outd[c] = ( rz1 + bind[bc-1] + bind[bc-bi] + r0 +
				bind[bc+bi] + bind[bc+1] + rz2 ) / (T) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = __ldg(&ind[c+ij]);
	}
}
#endif
/// END KERNELS

/// BEGIN HOST WRAPPERS

cudaError_t cuda_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data1, void * data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], void * reduced_val,
		meta_type_id type, int async, cudaEvent_t ((*event)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_len;
	dim3 grid;
	dim3 block;
	int iters;
	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		block = METAMORPH_CUDA_DEFAULT_BLOCK;
		//do not subtract 1 from blocx for the case when end == start
		grid = dim3((((*arr_end)[0] - (*arr_start)[0] + (block.x)) / block.x),
				(((*arr_end)[1] - (*arr_start)[1] + (block.y)) / block.y), 1);
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block.z)) / block.z);
	} else {
		grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
		block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
		iters = (int) (*grid_size)[2];
	}
	smem_len = block.x * block.y * block.z;
	if (event != NULL) {
		cudaEventCreate(&(*event)[0]);
		cudaEventRecord((*event)[0], 0);
	}
	switch (type) {
	case a_db:
		kernel_dotProd<double><<<grid,block,smem_len*sizeof(double)>>>((double*)data1, (double*)data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (double*)reduced_val, smem_len);
		break;

	case a_fl:
		kernel_dotProd<float><<<grid,block,smem_len*sizeof(float)>>>((float*)data1, (float*)data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (float*)reduced_val, smem_len);
		break;

	case a_ul:
		kernel_dotProd<unsigned long long><<<grid,block,smem_len*sizeof(unsigned long long)>>>((unsigned long long*)data1, (unsigned long long*)data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (unsigned long long*)reduced_val, smem_len);
		break;

	case a_in:
		kernel_dotProd<int><<<grid,block,smem_len*sizeof(int)>>>((int*)data1, (int*)data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (int *)reduced_val, smem_len);
		break;

	case a_ui:
		kernel_dotProd<unsigned int><<<grid,block,smem_len*sizeof(unsigned int)>>>((unsigned int*)data1, (unsigned int*)data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (unsigned int*)reduced_val, smem_len);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'cuda_dotProd' not implemented for selected type!\n");
		break;
	}
	if (event != NULL) {
		cudaEventCreate(&(*event)[1]);
		cudaEventRecord((*event)[1], 0);
	}
	if (!async)
		ret = cudaThreadSynchronize();
	return (ret);
}

cudaError_t cuda_reduce(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data, size_t (*array_size)[3], size_t (*arr_start)[3],
		size_t (*arr_end)[3], void * reduced_val, meta_type_id type, int async,
		cudaEvent_t ((*event)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_len;
	dim3 grid;
	dim3 block;
	int iters;
	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		block = METAMORPH_CUDA_DEFAULT_BLOCK;
		grid = dim3((((*arr_end)[0] - (*arr_start)[0] + (block.x)) / block.x),
				(((*arr_end)[1] - (*arr_start)[1] + (block.y)) / block.y), 1);
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block.z)) / block.z);
	} else {
		grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
		block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
		iters = (*grid_size)[2];
	}
	smem_len = block.x * block.y * block.z;
	if (event != NULL) {
		cudaEventCreate(&(*event)[0]);
		cudaEventRecord((*event)[0], 0);
	}
	//printf("CUDA Config: grid(%d, %d, %d) block(%d, %d, %d) iters %d\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, iters);
	switch (type) {
	case a_db:

		kernel_reduction3<double><<<grid,block,smem_len*sizeof(double)>>>((double*)data, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (double*)reduced_val, smem_len);
		break;

	case a_fl:
		kernel_reduction3<float><<<grid,block,smem_len*sizeof(float)>>>((float*)data, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (float*)reduced_val, smem_len);
		break;

	case a_ul:
		kernel_reduction3<unsigned long long><<<grid,block,smem_len*sizeof(unsigned long long)>>>((unsigned long long*)data, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (unsigned long long*)reduced_val, smem_len);
		break;

	case a_in:
		kernel_reduction3<int><<<grid,block,smem_len*sizeof(int)>>>((int*)data, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (int*)reduced_val, smem_len);
		break;

	case a_ui:
		kernel_reduction3<unsigned int><<<grid,block,smem_len*sizeof(unsigned int)>>>((unsigned int*)data, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, (unsigned int*)reduced_val, smem_len);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'cuda_reduce' not implemented for selected type!\n");
		break;

	}
	if (event != NULL) {
		cudaEventCreate(&(*event)[1]);
		cudaEventRecord((*event)[1], 0);
	}
	if (!async)
		ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return (ret);
}

cudaError_t cuda_transpose_face(size_t (*grid_size)[3],
		size_t (*block_size)[3], void *indata, void *outdata,
		size_t (*arr_dim_xy)[3], size_t (*tran_dim_xy)[3], meta_type_id type,
		int async, cudaEvent_t ((*event)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_len;
	dim3 grid, block;
//	size_t smem_len = (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
//TODO: Update to actually use user-provided grid/block once multi-element-per-thread
// scaling is added
//	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
//	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
	//FIXME: make this smart enough to rescale the threadblock (and thus shared memory - e.g. bank conflicts) w.r. double vs. float
	if (grid_size == NULL || block_size == NULL) {
		//FIXME: reconcile TILE_DIM/BLOCK_ROWS
		block = dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, 1);
		grid = dim3(((*tran_dim_xy)[0] + block.x - 1) / block.x,
				((*tran_dim_xy)[1] + block.y - 1) / block.y, 1);

	} else {
		grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
		block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
	}
	//The +1 here is to avoid bank conflicts with 32 floats or 16 doubles and is required by the kernel logic
	smem_len = (block.x + 1) * block.y * block.z;
	if (event != NULL) {
		cudaEventCreate(&(*event)[0]);
		cudaEventRecord((*event)[0], 0);
	}
	switch (type) {
	case a_db:
		kernel_transpose_2d<double><<<grid, block, smem_len*sizeof(double)>>>((double*)outdata, (double*)indata, (*arr_dim_xy)[0], (*arr_dim_xy)[1], (*tran_dim_xy)[0], (*tran_dim_xy)[1]);
		break;

	case a_fl:
		kernel_transpose_2d<float><<<grid, block, smem_len*sizeof(float)>>>((float*)outdata, (float*)indata, (*arr_dim_xy)[0], (*arr_dim_xy)[1], (*tran_dim_xy)[0], (*tran_dim_xy)[1]);
		break;

	case a_ul:
		kernel_transpose_2d<unsigned long><<<grid, block, smem_len*sizeof(unsigned long)>>>((unsigned long*)outdata, (unsigned long*)indata, (*arr_dim_xy)[0], (*arr_dim_xy)[1], (*tran_dim_xy)[0], (*tran_dim_xy)[1]);
		break;

	case a_in:
		kernel_transpose_2d<int><<<grid, block, smem_len*sizeof(int)>>>((int*)outdata, (int*)indata, (*arr_dim_xy)[0], (*arr_dim_xy)[1], (*tran_dim_xy)[0], (*tran_dim_xy)[1]);
		break;

	case a_ui:
		kernel_transpose_2d<unsigned int><<<grid, block, smem_len*sizeof(unsigned int)>>>((unsigned int*)outdata, (unsigned int*)indata, (*arr_dim_xy)[0], (*arr_dim_xy)[1], (*tran_dim_xy)[0], (*tran_dim_xy)[1]);
		break;

	default:
		fprintf(stderr,
				"Error: function 'cuda_transpose_face' not implemented for selected type!\n");
		break;
	}
	if (event != NULL) {
		cudaEventCreate(&(*event)[1]);
		cudaEventRecord((*event)[1], 0);
	}
	if (!async)
		ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return (ret);
}

cudaError_t cuda_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async,
		cudaEvent_t ((*event_k1)[2]), cudaEvent_t ((*event_c1)[2]),
		cudaEvent_t ((*event_c2)[2]), cudaEvent_t ((*event_c3)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_size;
	dim3 grid, block;
//TODO: Update to actually use user-provided grid/block once multi-element-per-thread
// scaling is added
//	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
//	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);

	//copy required piece of the face struct into constant memory
	if (event_c1 != NULL) {
		cudaEventCreate(&(*event_c1)[0]);
		cudaEventRecord((*event_c1)[0], 0);
	}
	cudaMemcpyToSymbol(c_face_size, face->size, face->count * sizeof(int));
	if (event_c1 != NULL) {
		cudaEventCreate(&(*event_c1)[1]);
		cudaEventRecord((*event_c1)[1], 0);
	}

	if (event_c2 != NULL) {
		cudaEventCreate(&(*event_c2)[0]);
		cudaEventRecord((*event_c2)[0], 0);
	}
	cudaMemcpyToSymbol(c_face_stride, face->stride, face->count * sizeof(int));
	if (event_c2 != NULL) {
		cudaEventCreate(&(*event_c2)[1]);
		cudaEventRecord((*event_c2)[1], 0);
	}

	if (event_c3 != NULL) {
		cudaEventCreate(&(*event_c3)[0]);
		cudaEventRecord((*event_c3)[0], 0);
	}
	cudaMemcpyToSymbol(c_face_child_size, remain_dim,
			face->count * sizeof(int));
	if (event_c3 != NULL) {
		cudaEventCreate(&(*event_c3)[1]);
		cudaEventRecord((*event_c3)[1], 0);
	}
//TODO: Create a grid/block similar to Kaixi's look at mpi_wrap.c to figure out how size is computed
	if (event_k1 != NULL) {
		cudaEventCreate(&(*event_k1)[0]);
		cudaEventRecord((*event_k1)[0], 0);
	}
	//FIXME: specify a unique macro for each default blocksize
	if (grid_size == NULL || block_size == NULL) {
		block = dim3(256, 1, 1);
		grid = dim3(
				(face->size[0] * face->size[1] * face->size[2] + block.x - 1)
						/ block.x, 1, 1);
	} else {
		//This is a workaround for some non-determinism that was observed when allowing fully-arbitrary spec of grid/block
		if ((*block_size)[1] != 1 || (*block_size)[2] != 1
				|| (*grid_size)[1] != 1 || (*grid_size)[2])
			fprintf(stderr,
					"WARNING: Pack requires 1D block and grid, ignoring y/z params!\n");
		//block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
		block = dim3((*block_size)[0], 1, 1);
		//grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
		grid = dim3((*grid_size)[0], 1, 1);
	}
	smem_size = face->count * block.x * sizeof(int);
	switch (type) {
	case a_db:
		kernel_pack<double><<<grid, block, smem_size>>>((double *)packed_buf, (double *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_fl:
		kernel_pack<float><<<grid, block, smem_size>>>((float *)packed_buf, (float *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_ul:
		kernel_pack<unsigned long><<<grid, block, smem_size>>>((unsigned long *)packed_buf, (unsigned long *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_in:
		kernel_pack<int><<<grid, block, smem_size>>>((int *)packed_buf, (int *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_ui:
		kernel_pack<unsigned int><<<grid, block, smem_size>>>((unsigned int *)packed_buf, (unsigned int *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	default:
		fprintf(stderr,
				"Error: function 'cuda_pack_face' not implemented for selected type!\n");
		break;
	}
	if (event_k1 != NULL) {
		cudaEventCreate(&(*event_k1)[1]);
		cudaEventRecord((*event_k1)[1], 0);
	}
	if (!async)
		ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return (ret);

}

cudaError_t cuda_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async,
		cudaEvent_t ((*event_k1)[2]), cudaEvent_t ((*event_c1)[2]),
		cudaEvent_t ((*event_c2)[2]), cudaEvent_t ((*event_c3)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_size;
	dim3 grid, block;
//TODO: Update to actually use user-provided grid/block once multi-element-per-thread
// scaling is added
//	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
//	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);

	//copy required piece of the face struct into constant memory
	if (event_c1 != NULL) {
		cudaEventCreate(&(*event_c1)[0]);
		cudaEventRecord((*event_c1)[0], 0);
	}
	cudaMemcpyToSymbol(c_face_size, face->size, face->count * sizeof(int));
	if (event_c1 != NULL) {
		cudaEventCreate(&(*event_c1)[1]);
		cudaEventRecord((*event_c1)[1], 0);
	}

	if (event_c2 != NULL) {
		cudaEventCreate(&(*event_c2)[0]);
		cudaEventRecord((*event_c2)[0], 0);
	}
	cudaMemcpyToSymbol(c_face_stride, face->stride, face->count * sizeof(int));
	if (event_c2 != NULL) {
		cudaEventCreate(&(*event_c2)[1]);
		cudaEventRecord((*event_c2)[1], 0);
	}

	if (event_c3 != NULL) {
		cudaEventCreate(&(*event_c3)[0]);
		cudaEventRecord((*event_c3)[0], 0);
	}
	cudaMemcpyToSymbol(c_face_child_size, remain_dim,
			face->count * sizeof(int));
	if (event_c3 != NULL) {
		cudaEventCreate(&(*event_c3)[1]);
		cudaEventRecord((*event_c3)[1], 0);
	}
//TODO: Create a grid/block similar to Kaixi's look at mpi_wrap.c to figure out how size is computed
	if (event_k1 != NULL) {
		cudaEventCreate(&(*event_k1)[0]);
		cudaEventRecord((*event_k1)[0], 0);
	}
	//FIXME: specify a unique macro for each default blocksize
	if (grid_size == NULL || block_size == NULL) {
		block = dim3(256, 1, 1);
		grid = dim3(
				(face->size[0] * face->size[1] * face->size[2] + block.x - 1)
						/ block.x, 1, 1);
	} else {
		//This is a workaround for some non-determinism that was observed when allowing fully-arbitrary spec of grid/block
		if ((*block_size)[1] != 1 || (*block_size)[2] != 1
				|| (*grid_size)[1] != 1 || (*grid_size)[2])
			fprintf(stderr,
					"WARNING: Unpack requires 1D block and grid, ignoring y/z params!\n");
		//block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
		block = dim3((*block_size)[0], 1, 1);
		//grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
		grid = dim3((*grid_size)[0], 1, 1);
	}
	smem_size = face->count * block.x * sizeof(int);
	switch (type) {
	case a_db:
		kernel_unpack<double><<<grid, block, smem_size>>>((double *)packed_buf, (double *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_fl:
		kernel_unpack<float><<<grid, block, smem_size>>>((float *)packed_buf, (float *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_ul:
		kernel_unpack<unsigned long><<<grid, block, smem_size>>>((unsigned long *)packed_buf, (unsigned long *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_in:
		kernel_unpack<int><<<grid, block, smem_size>>>((int *)packed_buf, (int *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	case a_ui:
		kernel_unpack<unsigned int><<<grid, block, smem_size>>>((unsigned int *)packed_buf, (unsigned int *)buf, face->size[0]*face->size[1]*face->size[2], face->start, face->count);
		break;

	default:
		fprintf(stderr,
				"Error: function 'cuda_unpack_face' not implemented for selected type!\n");
		break;
	}
	if (event_k1 != NULL) {
		cudaEventCreate(&(*event_k1)[1]);
		cudaEventRecord((*event_k1)[1], 0);
	}
	if (!async)
		ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return (ret);

}

cudaError_t cuda_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * indata, void * outdata, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], meta_type_id type,
		int async, cudaEvent_t ((*event)[2])) {
	cudaError_t ret = cudaSuccess;
	size_t smem_len;
	dim3 grid;
	dim3 block;
	int iters;
	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		//block = METAMORPH_CUDA_DEFAULT_BLOCK;
		block = dim3(64, 4, 1);
		//do not subtract 1 from blocx for the case when end == start
		grid = dim3((((*arr_end)[0] - (*arr_start)[0] + (block.x)) / block.x),
				(((*arr_end)[1] - (*arr_start)[1] + (block.y)) / block.y), 1);
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block.z)) / block.z);
	} else {
		grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
		block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
		iters = (int) (*grid_size)[2];
	}
	smem_len = (block.x + 2) * (block.y + 2) * block.z;
	//smem_len = 0;

	if (event != NULL) {
		cudaEventCreate(&(*event)[0]);
		cudaEventRecord((*event)[0], 0);
	}

	switch (type) {
	case a_db:
		kernel_stencil_3d7p<double><<<grid,block,smem_len*sizeof(double)>>>((double*)indata, (double*)outdata, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, smem_len);
		break;

	case a_fl:
		kernel_stencil_3d7p<float><<<grid,block,smem_len*sizeof(float)>>>((float*)indata, (float*)outdata, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, smem_len);
		break;

	case a_ul:
		kernel_stencil_3d7p<unsigned long long><<<grid,block,smem_len*sizeof(unsigned long long)>>>((unsigned long long*)indata, (unsigned long long*)outdata, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, smem_len);
		break;

	case a_in:
		kernel_stencil_3d7p<int><<<grid,block,smem_len*sizeof(int)>>>((int*)indata, (int*)outdata, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, smem_len);
		break;

	case a_ui:
		kernel_stencil_3d7p<unsigned int><<<grid,block,smem_len*sizeof(unsigned int)>>>((unsigned int*)indata, (unsigned int*)outdata, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[0], (*arr_end)[1], (*arr_end)[2], iters, smem_len);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'cuda_stencil_3d7p' not implemented for selected type!\n");
		break;
	}

	if (event != NULL) {
		cudaEventCreate(&(*event)[1]);
		cudaEventRecord((*event)[1], 0);
	}
	if (!async)
		ret = cudaThreadSynchronize();
	return (ret);

}
