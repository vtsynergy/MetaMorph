/** \file
 * CUDA Back-End kernel and wrapper implementations
 */

#include <stdio.h>

#include "mm_cuda_backend.cuh"
#include "metamorph_dynamic_symbols.h"


/** Reuse the function pointers from the profiling plugin if it's found */
extern struct profiling_dyn_ptrs profiling_symbols;
/** non-specialized shared memory class template */
template<typename T>
class SharedMem {
public:
	// Ensure that we won't compile any un-specialized types
	__device__
	/** Just get the address of the shared memory region
	 * \return The start address of the shared memory region
	 */
	T* getPointer() {
		return (T*) NULL;
	}
	;
};

/** specialization for double */
template<>
class SharedMem<double> {
public:
	__device__
	/** Just get the address of the shared memory region
	 * \return The start address of the shared memory region
	 */
	double* getPointer() {
		extern __shared__ double s_db[]; return s_db;
	}
};

/** specialization for float */
template<>
class SharedMem<float> {
public:
	__device__
	/** Just get the address of the shared memory region
	 * \return The start address of the shared memory region
	 */
	float* getPointer() {
		extern __shared__ float s_fl[]; return s_fl;
	}
};

/** specialization for unsigned long long */
template<>
class SharedMem<unsigned long long> {
public:
	__device__
	/** Just get the address of the shared memory region
	 * \return The start address of the shared memory region
	 */
	unsigned long long* getPointer() {
		extern __shared__ unsigned long long s_ul[]; return s_ul;
	}
};

/** specialization for int */
template<>
class SharedMem<int> {
public:
	__device__
	/** Just get the address of the shared memory region
	 * \return The start address of the shared memory region
	 */
	int* getPointer() {
		extern __shared__ int s_in[]; return s_in;
	}
};

/** specialization for unsigned int */
template<>
class SharedMem<unsigned int> {
public:
	__device__
	/** Just get the address of the shared memory region
	 * \return The start address of the shared memory region
	 */
	unsigned int* getPointer() {
		extern __shared__ unsigned int s_ui[]; return s_ui;
	}
};

/** A generic in-block tree reduction-sum.
 * Does not relax synchronization once inside a warp.
 * \param psum a pointer to a shared memory array for storing partial sums
 * \param tid my thread id
 * \param len_ number of threads in the block
 */
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

/** Implementation of double atomicAdd from CUDA Programming Guide: Appendix B.12.
 * \param address the read-write address
 * \param val the value to add
 * \return the old value at address after successfully writing
 * \todo TODO figure out how to use templates with the __X_as_Y intrinsics (???)
 */
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

/**
 * wrapper for all the types natively supported by CUDA
 * \param addr the read/write address
 * \param val the value to add
 * \return the previous value of addr
*/
template <typename T> __device__ T atomicAdd_wrapper(T* addr, T val) {
	return atomicAdd(addr, val);
}

/**
 * wrapper for hand-implemented double type
 * \param addr the read/write address
 * \param val the value to add
 * \return the previous value of addr
*/
template <> __device__ double atomicAdd_wrapper<double>(double* addr, double val) {
	return atomicAdd(addr, val);
}

/**
 * 3D dot-product kernel with bound control.
 * s* and e* are only necessary when the halo layers 
 *  have different thicknesses along various directions.
 * \param phi1 first input array
 * \param phi2 second input array
 * \param sx start index in X dimension
 * \param sy start index in Y dimension
 * \param sz start index in Z dimension
 * \param ex end index in X dimension
 * \param ey end index in Y dimension
 * \param ez end index in Z dimension
 * \param i phi1 and phi2's X dimension
 * \param j phi1 and phi2's Y dimension
 * \param k phi1 and phi2's Z dimension
 * \param gz how many times to loop over the Z dimension to account for 2D grid
 * \param len_  number of threads in a threadblock.
 * \param reduction storage for the globally-reduced final value
 */
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

/**
 * 3D reduction-sum kernel with bound control.
 * s* and e* are only necessary when the halo layers 
 *  have different thicknesses along various directions.
 * \param phi input array to reduce
 * \param sx start index in X dimension
 * \param sy start index in Y dimension
 * \param sz start index in Z dimension
 * \param ex end index in X dimension
 * \param ey end index in Y dimension
 * \param ez end index in Z dimension
 * \param i phi's X dimension
 * \param j phi's Y dimension
 * \param k phi's Z dimension
 * \param gz how many times to loop over the Z dimension to account for 2D grid
 * \param len_  number of threads in a threadblock.
 * \param reduction storage for the globally-reduced final value
 */
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
/** The number of elements in each level of the face struct */
__constant__ int c_face_size[METAMORPH_FACE_MAX_DEPTH];
/** The stride between elements at each level of the face struct */
__constant__ int c_face_stride[METAMORPH_FACE_MAX_DEPTH];
/** Size of all children (>= level+1) so at level 0, child_size = total_num_face_elements */
__constant__ int c_face_child_size[METAMORPH_FACE_MAX_DEPTH];

/** Helper function to compute the integer read offset for buffer packing
 * \param tid my thread ID
 * \param a shared memory region used to compute partial offsets
 * \param start initial global offset
 * \param count the number of layers in the face structure (usually the number of dimensions of the buffer)
 * \return the offset index for this thread to read/write to/from
 * \todo TODO: Add support for multi-dimensional grid/block
*/
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

/**
 * A kernel to pack a subregion of a 3D buffer into contiguous memory
 * \param packed_buf the output buffer, of sufficient size to store an entire face/slab
 * \param buf the unpacked 3D buffer
 * \param size the total number of elements in packed_buf
 * \param start the initial offset in buf to start computing from
 * \param count the number of layers in the face struct stored in constant memory
 */
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

/**
 * A kernel to unpack a subregion of a 3D buffer into contiguous memory
 * \param packed_buf the input buffer, of sufficient size to store an entire face/slab
 * \param buf the unpacked 3D buffer
 * \param size the total number of elements in packed_buf
 * \param start the initial offset in buf to start computing from
 * \param count the number of layers in the face struct stored in constant memory
 */
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

/** Kernel to transpose a 2D (sub-)region of a 2D array
 * \param odata the output transposed 2D array (of size tran_width * tran_height)
 * \param idata the input untransposed 2D array (of size arr_widt * arr_height)
 * \param arr_width the width of the input array
 * \param arr_height the height of the input array
 * \param tran_width the width of the output array
 * \param tran_height the height of the output array
 */
template <typename T>
__global__ void kernel_transpose_2d(T *odata, T *idata, int arr_width, int arr_height, int tran_width, int tran_height)
{
	SharedMem<T> shared;
	T * tile = shared.getPointer();

	int blockIdx_x, blockIdx_y;
	int gridDim_x, gridDim_y;

// do diagonal reordering
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

		int xIndex_in = blockIdx_x * blockDim.x + threadIdx.x;
		int yIndex_in = blockIdx_y * blockDim.y + threadIdx.y;
		int index_in = xIndex_in + (yIndex_in)*arr_width;

		int xIndex_out = blockIdx_y * blockDim.y + threadIdx.x;
		int yIndex_out = blockIdx_x * blockDim.x + threadIdx.y;
		int index_out = xIndex_out + (yIndex_out)*arr_height;

		//The blockDim.x+1 in R/W steps is to avoid bank conflicts if blockDim.x==16 or 32
		if(xIndex_in < tran_width && yIndex_in < tran_height)
		tile[threadIdx.x+(blockDim.x+1)*threadIdx.y] = idata[index_in];

		__syncthreads();

		if(xIndex_out < tran_height && yIndex_out < tran_width)
		odata[index_out] = tile[threadIdx.y+(blockDim.y+1)*threadIdx.x];

		//Added when the loop was added to ensure writes are finished before new vals go into shared memory
		__syncthreads();
	}
}

/** A kernel to compute a 3D 7-point averaging stencil
 * (i.e sum the cubic cell and its 6 directly-adjacent neighbors and divide by 7)
 * Optimizations: Read-only cache + Register blocking (Z) + smem blocking (X-Y)
 * \param ind a non-aliased 3D region of size (i*j*k)
 * \param outd a non-aliased 3D region of size (i*j*k)
 * \param i size of ind and outd in the x dimension
 * \param j size of ind and outd in the y dimension
 * \param k size of ind and outd in the z dimension
 * \param sx index of starting halo cell in the x dimension
 * \param sy index of starting halo cell in the y dimension
 * \param sz index of starting halo cell in the z dimension
 * \param ex index of ending halo cell in the x dimension
 * \param ey index of ending halo cell in the y dimension
 * \param ez index of ending halo cell in the z dimension
 * \param gz number of global z iterations to compute to account for 2D grid
 * \param len_ number of threads in a threadblock
 * \warning assumes that s* and e* bounds include a 1-thick halo
 *   (i.e. will only compute values for cells in T([sx+1:ex-1], [sy+1:ey-1], [sz+1:ez-1])
 * \warning this kernel works for 3D data only.
 * \warning works only with 2D thread blocks (use rectangular blocks, i.e. 64*4, 128*2)
 */
template <typename T>
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

a_err metaCUDAAlloc(void ** ptr, size_t size) {
  return cudaMalloc(ptr, size);
}
a_err metaCUDAFree(void * ptr) {
  return cudaFree(ptr);
}
  a_err metaCUDAWrite(void * dst, void * src, size_t size, a_bool async, meta_callback * call, meta_event * ret_event) {
  a_err ret = cudaSuccess;
  cudaEvent_t * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferCUDA && ret_event->event_pl != NULL) {
    events = ((cudaEvent_t *)ret_event->event_pl);
  }
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferCUDA, size);
    if (events == NULL) {
      events = ((cudaEvent_t *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
	if (events != NULL) {
		cudaEventRecord(events[0], 0);
	}
		if (async) {
			ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
		} else {
			ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
		}
	if (events != NULL) {
		cudaEventRecord(events[1], 0);
	}
//FIXME CUDA needs to deal with its own callback setup
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, c_H2D);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
  return ret;
}
  a_err metaCUDARead(void * dst, void * src, size_t size, a_bool async, meta_callback * call, meta_event * ret_event) {
  a_err ret = cudaSuccess;
  cudaEvent_t * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferCUDA && ret_event->event_pl != NULL) {
    events = ((cudaEvent_t *)ret_event->event_pl);
  }
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferCUDA, size);
    if (events == NULL) {
      events = ((cudaEvent_t *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
	if (events != NULL) {
		cudaEventRecord(events[0], 0);
	}
		if (async) {
			ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, 0);
		} else {
			ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
		}
	if (events != NULL) {
		cudaEventRecord(events[1], 0);
	}
//FIXME CUDA needs to deal with its own callback setup
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, c_D2H);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
  return ret;
}
  a_err metaCUDADevCopy(void * dst, void * src, size_t size, a_bool async, meta_callback * call, meta_event * ret_event) {
  a_err ret = cudaSuccess;
  cudaEvent_t * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferCUDA && ret_event->event_pl != NULL) {
    events = ((cudaEvent_t *)ret_event->event_pl);
  }
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferCUDA, size);
    if (events == NULL) {
      events = ((cudaEvent_t *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
	if (events != NULL) {
		cudaEventRecord(events[0], 0);
	}
		if (async) {
			ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, 0);
		} else {
			ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
		}
	if (events != NULL) {
		cudaEventRecord(events[1], 0);
	}
//FIXME CUDA needs to deal with its own callback setup
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, c_D2D);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
  return ret;
}
a_err metaCUDAInitByID(a_int accel) {
  a_err ret = cudaSetDevice(accel);
  //Because of how CUDA destructs stuff automatically, we need to force MM to destruct before them (for profiling, any other forced flushes), which requires registering it with atexit *after* CUDA does, and cudaSetDevice is known to register destructor functions, so here we are re-registering our global destructor /shrug
  atexit(meta_finalize);
  return ret;
}
a_err metaCUDACurrDev(a_int * accel) {
  return cudaGetDevice(accel);
}
a_err metaCUDAMaxWorkSizes(a_dim3 * grid, a_dim3 * block) {
  fprintf(stderr, "metaCUDAMaxWorkSizes unimplemented\n");
  return -1;
}
a_err metaCUDAFlush() {
  return cudaThreadSynchronize();
}
a_err metaCUDACreateEvent(void** ret_event) {
  a_err ret = cudaSuccess;
  if (ret_event != NULL) {
    *ret_event = malloc(sizeof(cudaEvent_t)*2);
    ret = cudaEventCreate(&((cudaEvent_t *)(*ret_event))[0]);
    ret |= cudaEventCreate(&((cudaEvent_t *)(*ret_event))[1]);
  }
  else ret = cudaErrorInvalidValue;
  return ret;
}

a_err metaCUDADestroyEvent(void * event) {
  a_err ret = cudaSuccess;
  if (event != NULL) {
    ret = cudaEventDestroy(((cudaEvent_t*)event)[0]);
    ret |= cudaEventDestroy(((cudaEvent_t*)event)[1]);
    free(event);
  }
  else ret = cudaErrorInvalidValue;
  return ret;
}

a_err metaCUDAEventElapsedTime(float * ret_ms, meta_event event) {
  a_err ret = cudaSuccess;
  if (ret_ms != NULL && event.event_pl != NULL) {
    cudaEvent_t * events = (cudaEvent_t *)event.event_pl;
    ret = cudaEventElapsedTime(ret_ms, events[0], events[1]);
  }
  else ret = cudaErrorInvalidValue;
  return ret;
}
/**
 * This struct just allows the status and event values returned by the CUDA callback to be passed through
 * the meta_callback payload back up to the user
 */
struct cuda_callback_data {
  /** The returned status from a CUDA callback */
  cudaError_t status;
  /** The returned stream from a CUDA callback */
  cudaStream_t stream;
};

void CUDART_CB  metaCUDACallbackHelper(cudaStream_t stream, cudaError_t status, void * data) {
  if (data == NULL) return;
  meta_callback * payload = (meta_callback *) data;
  struct cuda_callback_data * info = (struct cuda_callback_data *)calloc(1, sizeof(struct cuda_callback_data));
  info->status = status;
  info->stream = stream;
  payload->backend_status = info;
  (payload->callback_func)((meta_callback *)data);
}

a_err metaCUDAExpandCallback(meta_callback call, cudaStream_t * ret_stream, cudaError_t * ret_status, void** ret_data) {
  if (call.backend_status == NULL || ret_status == NULL || ret_data == NULL || ret_stream == NULL) return cudaErrorInvalidValue;
  (*ret_stream) = ((struct cuda_callback_data *)call.backend_status)->stream;
  (*ret_status) = ((struct cuda_callback_data *)call.backend_status)->status;
  (*ret_data) = call.data_payload;
  return cudaSuccess;
}
a_err metaCUDARegisterCallback(meta_callback * call) {
  a_err ret = cudaSuccess;
    if (call == NULL || call->callback_func == NULL || call->data_payload == NULL) ret = cudaErrorInvalidValue;
    else {
      ret = cudaStreamAddCallback(0, metaCUDACallbackHelper, (void*)call, 0);
    }
return ret;
}


  a_err cuda_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], void * data1, void * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	a_err ret = cudaSuccess;
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
  cudaEvent_t * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferCUDA && ret_event->event_pl != NULL) {
    events = ((cudaEvent_t *)ret_event->event_pl);
  }
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferCUDA, (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type));
    if (events == NULL) {
      events = ((cudaEvent_t *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
	if (events != NULL) {
		cudaEventRecord(events[0], 0);
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
	if (events != NULL) {
		cudaEventRecord(events[1], 0);
	}
//FIXME CUDA needs to deal with its own callback setup
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_dotProd);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
	if (!async)
		ret = cudaThreadSynchronize();
	return (ret);
}

  a_err cuda_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], void * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	a_err ret = cudaSuccess;
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
  cudaEvent_t * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferCUDA && ret_event->event_pl != NULL) {
    events = ((cudaEvent_t *)ret_event->event_pl);
  }
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferCUDA, (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type));
    if (events == NULL) {
      events = ((cudaEvent_t *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
	if (events != NULL) {
		cudaEventRecord(events[0], 0);
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
	if (events != NULL) {
		cudaEventRecord(events[1], 0);
	}
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_reduce);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
	if (!async)
		ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return (ret);
}

  a_err cuda_transpose_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void * outdata, size_t (* arr_dim_xy)[3], size_t (* tran_dim_xy)[3], meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	a_err ret = cudaSuccess;
	size_t smem_len;
	dim3 grid, block;
	/// \todo FIXME: make this smart enough to rescale the threadblock (and thus shared memory - e.g. bank conflicts) w.r. double vs. float
	if (grid_size == NULL || block_size == NULL) {
		/// \todo FIXME: reconcile TILE_DIM/BLOCK_ROWS (???)
		block = dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, 1);
		grid = dim3(((*tran_dim_xy)[0] + block.x - 1) / block.x,
				((*tran_dim_xy)[1] + block.y - 1) / block.y, 1);

	} else {
		grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
		block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
	}
	/// \warning The +1 here is to avoid bank conflicts with 32 floats or 16 doubles and is required by the kernel logic
	smem_len = (block.x + 1) * block.y * block.z;
  cudaEvent_t * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferCUDA && ret_event->event_pl != NULL) {
    events = ((cudaEvent_t *)ret_event->event_pl);
  }
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferCUDA, (*tran_dim_xy)[0]*(*tran_dim_xy)[1]*get_atype_size(type));
    if (events == NULL) {
      events = ((cudaEvent_t *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
	if (events != NULL) {
		cudaEventRecord(events[0], 0);
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
	if (events != NULL) {
		cudaEventRecord(events[1], 0);
	}
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_transpose_2d_face);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
	if (!async)
		ret = cudaThreadSynchronize();
	return (ret);
}

  a_err cuda_pack_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * packed_buf, void * buf, meta_face * face, int * remain_dim, meta_type_id type, int async, meta_callback * call, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3) {
	a_err ret = cudaSuccess;
	size_t smem_size;
	dim3 grid, block;
//	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
//	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);

	//copy required piece of the face struct into constant memory
  cudaEvent_t * events_c1 = NULL;
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferCUDA && ret_event_c1->event_pl != NULL) {
    events_c1 = ((cudaEvent_t *)ret_event_c1->event_pl);
  }
  meta_timer * timer_c1 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c1, metaModePreferCUDA, get_atype_size(type)*3);
    if (events_c1 == NULL) {
      events_c1 = ((cudaEvent_t *)timer_c1->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_c1->event = *ret_event_c1;
    }
  }
	if (events_c1 != NULL) {
		cudaEventRecord(events_c1[0], 0);
	}
	cudaMemcpyToSymbol(c_face_size, face->size, face->count * sizeof(int));
	if (events_c1 != NULL) {
		cudaEventRecord(events_c1[1], 0);
	}

  cudaEvent_t * events_c2 = NULL;
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferCUDA && ret_event_c2->event_pl != NULL) {
    events_c2 = ((cudaEvent_t *)ret_event_c2->event_pl);
  }
  meta_timer * timer_c2 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c2, metaModePreferCUDA, get_atype_size(type)*3);
    if (events_c2 == NULL) {
      events_c2 = ((cudaEvent_t *)timer_c2->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_c2->event = *ret_event_c2;
    }
  }
	if (events_c2 != NULL) {
		cudaEventRecord(events_c2[0], 0);
	}
	cudaMemcpyToSymbol(c_face_stride, face->stride, face->count * sizeof(int));
	if (events_c2 != NULL) {
		cudaEventRecord(events_c2[1], 0);
	}

  cudaEvent_t * events_c3 = NULL;
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferCUDA && ret_event_c3->event_pl != NULL) {
    events_c3 = ((cudaEvent_t *)ret_event_c3->event_pl);
  }
  meta_timer * timer_c3 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c3, metaModePreferCUDA, get_atype_size(type)*3);
    if (events_c3 == NULL) {
      events_c3 = ((cudaEvent_t *)timer_c3->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_c3->event = *ret_event_c3;
    }
  }
	if (events_c3 != NULL) {
		cudaEventRecord(events_c3[0], 0);
	}
	cudaMemcpyToSymbol(c_face_child_size, remain_dim,
			face->count * sizeof(int));
	if (events_c3 != NULL) {
		cudaEventRecord(events_c3[1], 0);
	}
	/// \todo FIXME: specify a unique macro for each default blocksize
	if (grid_size == NULL || block_size == NULL) {
		block = dim3(256, 1, 1);
		grid = dim3(
				(face->size[0] * face->size[1] * face->size[2] + block.x - 1)
						/ block.x, 1, 1);
	} else {
		/// \warning This is a workaround for some non-determinism that was observed when allowing fully-arbitrary spec of grid/block
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
  cudaEvent_t * events_k1 = NULL;
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferCUDA && ret_event_k1->event_pl != NULL) {
    events_k1 = ((cudaEvent_t *)ret_event_k1->event_pl);
  }
  meta_timer * timer_k1 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_k1, metaModePreferCUDA, get_atype_size(type)*face->size[0]*face->size[1]*face->size[2]);
    if (events_k1 == NULL) {
      events_k1 = ((cudaEvent_t *)timer_k1->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_k1->event = *ret_event_k1;
    }
  }
	if (events_k1 != NULL) {
		cudaEventRecord(events_k1[0], 0);
	}
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
	if (events_k1 != NULL) {
		cudaEventRecord(events_k1[1], 0);
	}
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_k1, k_pack_2d_face);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c1, c_H2Dc);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c2, c_H2Dc);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c3, c_H2Dc);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
	if (!async)
		ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return (ret);

}

  a_err cuda_unpack_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * packed_buf, void * buf, meta_face * face, int * remain_dim, meta_type_id type, int async, meta_callback * call, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3) {
	a_err ret = cudaSuccess;
	size_t smem_size;
	dim3 grid, block;
//TODO: Update to actually use user-provided grid/block once multi-element-per-thread
// scaling is added
//	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
//	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);

	//copy required piece of the face struct into constant memory
  cudaEvent_t * events_c1 = NULL;
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferCUDA && ret_event_c1->event_pl != NULL) {
    events_c1 = ((cudaEvent_t *)ret_event_c1->event_pl);
  }
  meta_timer * timer_c1 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c1, metaModePreferCUDA, get_atype_size(type)*3);
    if (events_c1 == NULL) {
      events_c1 = ((cudaEvent_t *)timer_c1->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_c1->event = *ret_event_c1;
    }
  }
	if (events_c1 != NULL) {
		cudaEventRecord(events_c1[0], 0);
	}
	cudaMemcpyToSymbol(c_face_size, face->size, face->count * sizeof(int));
	if (events_c1 != NULL) {
		cudaEventRecord(events_c1[1], 0);
	}

  cudaEvent_t * events_c2 = NULL;
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferCUDA && ret_event_c2->event_pl != NULL) {
    events_c2 = ((cudaEvent_t *)ret_event_c2->event_pl);
  }
  meta_timer * timer_c2 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c2, metaModePreferCUDA, get_atype_size(type)*3);
    if (events_c2 == NULL) {
      events_c2 = ((cudaEvent_t *)timer_c2->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_c2->event = *ret_event_c2;
    }
  }
	if (events_c2 != NULL) {
		cudaEventRecord(events_c2[0], 0);
	}
	cudaMemcpyToSymbol(c_face_stride, face->stride, face->count * sizeof(int));
	if (events_c2 != NULL) {
		cudaEventRecord(events_c2[1], 0);
	}

  cudaEvent_t * events_c3 = NULL;
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferCUDA && ret_event_c3->event_pl != NULL) {
    events_c3 = ((cudaEvent_t *)ret_event_c3->event_pl);
  }
  meta_timer * timer_c3 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c3, metaModePreferCUDA, get_atype_size(type)*3);
    if (events_c3 == NULL) {
      events_c3 = ((cudaEvent_t *)timer_c3->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_c3->event = *ret_event_c3;
    }
  }
	if (events_c3 != NULL) {
		cudaEventRecord(events_c3[0], 0);
	}
	cudaMemcpyToSymbol(c_face_child_size, remain_dim,
			face->count * sizeof(int));
	if (events_c3 != NULL) {
		cudaEventRecord(events_c3[1], 0);
	}
//TODO: Create a grid/block similar to Kaixi's look at mpi_wrap.c to figure out how size is computed
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
  cudaEvent_t * events_k1 = NULL;
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferCUDA && ret_event_k1->event_pl != NULL) {
    events_k1 = ((cudaEvent_t *)ret_event_k1->event_pl);
  }
  meta_timer * timer_k1 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_k1, metaModePreferCUDA, get_atype_size(type)*face->size[0]*face->size[1]*face->size[2]);
    if (events_k1 == NULL) {
      events_k1 = ((cudaEvent_t *)timer_k1->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer_k1->event = *ret_event_k1;
    }
  }
	if (events_k1 != NULL) {
		cudaEventRecord(events_k1[0], 0);
	}
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
	if (events_k1 != NULL) {
		cudaEventRecord(events_k1[1], 0);
	}
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_k1, k_unpack_2d_face);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c1, c_H2Dc);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c2, c_H2Dc);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c3, c_H2Dc);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
	if (!async)
		ret = cudaThreadSynchronize();
	//TODO consider derailing for an explictly 2D/1D reduce..
	return (ret);

}

  a_err cuda_stencil_3d7p(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void * outdata, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	a_err ret = cudaSuccess;
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
  cudaEvent_t * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferCUDA && ret_event->event_pl != NULL) {
    events = ((cudaEvent_t *)ret_event->event_pl);
  }
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferCUDA, (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type));
    if (events == NULL) {
      events = ((cudaEvent_t *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created cudaEvent_t here since the profiling function calls create?
      //metaCUDADestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
	if (events != NULL) {
		cudaEventRecord(events[0], 0);
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
	if (events != NULL) {
		cudaEventRecord(events[1], 0);
	}
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL) cudaStreamAddCallback(0, metaCUDACallbackHelper, call, 0);
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_stencil_3d7p);
	//TODO if we do event copy, assign it back to the callbnack/ret_event here
	if (!async)
		ret = cudaThreadSynchronize();
	return (ret);

}
