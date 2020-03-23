/** \file
 * OpenCL Back-End Kernel implementations
*/

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#ifndef SINGLE_KERNEL_PROGS
#define DOUBLE
#define FLOAT
#define UNSIGNED_LONG
#define INTEGER
#define UNSIGNED_INTEGER
#define KERNEL_REDUCE
#define KERNEL_DOT_PROD
#define KERNEL_TRANSPOSE
#define KERNEL_PACK
#define KERNEL_UNPACK
#define KERNEL_STENCIL
#define KERNEL_CSR
#define KERNEL_CRC
#endif

#if (defined(DOUBLE) && (defined(KERNEL_REDUCE) || defined(KERNEL_DOT_PROD)))
/**
 * In-workgroup sum reduction on double-precision values
 * \param psum An array of at least _len initialized values to accumulate into the 0th cell
 * \param tid The id of the currently-running thread's accumulate cell
 * \param len_ the length of psum (number of values to reduce) 
 */
void block_reduction_db(__local volatile double *psum, int tid, int len_) {
	int stride = len_ >> 1;
	while (stride > 0) {
		if (tid < stride) psum[tid] += psum[tid+stride];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		stride >>= 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop from (stride >32) to ensure compatibility with CPU platforms.
	// once I can work out how to use the preferred_work_group_size_multiple that Tom suggested, I'll re-add the optimized version.   
	/*if (tid < 32) { 
	 psum[tid] += psum[tid+32];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+16];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+8];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+4];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+2];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+1];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 } */
}
#endif

#if (defined(FLOAT) && (defined(KERNEL_REDUCE) || defined(KERNEL_DOT_PROD)))
/**
 * In-workgroup sum reduction on single-precision values
 * \param psum An array of at least _len initialized values to accumulate into the 0th cell
 * \param tid The id of the currently-running thread's accumulate cell
 * \param len_ the length of psum (number of values to reduce) 
 */
void block_reduction_fl(__local volatile float *psum, int tid, int len_) {
	int stride = len_ >> 1;
	while (stride > 0) {
		if (tid < stride) psum[tid] += psum[tid+stride];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		stride >>= 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop from (stride >32) to ensure compatibility with CPU platforms.
	// once I can work out how to use the preferred_work_group_size_multiple that Tom suggested, I'll re-add the optimized version.   
	/*if (tid < 32) { 
	 psum[tid] += psum[tid+32];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+16];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+8];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+4];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+2];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+1];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 } */
}
#endif

#if (defined(UNSIGNED_LONG) && (defined(KERNEL_REDUCE) || defined(KERNEL_DOT_PROD)))
/**
 * In-workgroup sum reduction on unsigned long values
 * \param psum An array of at least _len initialized values to accumulate into the 0th cell
 * \param tid The id of the currently-running thread's accumulate cell
 * \param len_ the length of psum (number of values to reduce) 
 */
void block_reduction_ul(__local volatile unsigned long *psum, int tid, int len_) {
	int stride = len_ >> 1;
	while (stride > 0) {
		if (tid < stride) psum[tid] += psum[tid+stride];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		stride >>= 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop from (stride >32) to ensure compatibility with CPU platforms.
	// once I can work out how to use the preferred_work_group_size_multiple that Tom suggested, I'll re-add the optimized version.   
	/*if (tid < 32) { 
	 psum[tid] += psum[tid+32];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+16];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+8];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+4];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+2];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+1];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 } */
}
#endif

#if (defined(INTEGER) && (defined(KERNEL_REDUCE) || defined(KERNEL_DOT_PROD)))
/**
 * In-workgroup sum reduction on integer values
 * \param psum An array of at least _len initialized values to accumulate into the 0th cell
 * \param tid The id of the currently-running thread's accumulate cell
 * \param len_ the length of psum (number of values to reduce) 
 */
void block_reduction_in(__local volatile int *psum, int tid, int len_) {
	int stride = len_ >> 1;
	while (stride > 0) {
		if (tid < stride) psum[tid] += psum[tid+stride];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		stride >>= 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop from (stride >32) to ensure compatibility with CPU platforms.
	// once I can work out how to use the preferred_work_group_size_multiple that Tom suggested, I'll re-add the optimized version.   
	/*if (tid < 32) { 
	 psum[tid] += psum[tid+32];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+16];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+8];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+4];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+2];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+1];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 } */
}
#endif

#if (defined(UNSIGNED_INTEGER) && (defined(KERNEL_REDUCE) || defined(KERNEL_DOT_PROD)))
/**
 * In-workgroup sum reduction on unsigned integer values
 * \param psum An array of at least _len initialized values to accumulate into the 0th cell
 * \param tid The id of the currently-running thread's accumulate cell
 * \param len_ the length of psum (number of values to reduce) 
 */
void block_reduction_ui(__local volatile unsigned int *psum, int tid, int len_) {
	int stride = len_ >> 1;
	while (stride > 0) {
		if (tid < stride) psum[tid] += psum[tid+stride];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		stride >>= 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop from (stride >32) to ensure compatibility with CPU platforms.
	// once I can work out how to use the preferred_work_group_size_multiple that Tom suggested, I'll re-add the optimized version.   
	/*if (tid < 32) { 
	 psum[tid] += psum[tid+32];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+16];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+8];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+4];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+2];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 psum[tid] += psum[tid+1];
	 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	 } */
}
#endif

//Paul - Implementation of double atomicAdd from CUDA Programming Guide: Appendix B.12
// ported to OpenCL
#if (defined(DOUBLE) && (defined(KERNEL_REDUCE) || defined(KERNEL_DOT_PROD)))
/**
 * Manual implementation of a double-precision atomic add based on an atomic compare and exchange
 * \param address The location of the initialized double precision value to accumulate to
 * \param val The value to add to that already stored in address
 * \return The old value at address that val was added to
 */
double atomicAdd_db(__global double* address, double val)
{
	__global unsigned long * address_as_ul =
	(__global unsigned long *)address;
	//unsigned long old = *address_as_ul, assumed;
	volatile unsigned long old = atom_add(address_as_ul, 0), assumed;
	do {
		assumed = old;
		old = atom_cmpxchg(address_as_ul, assumed,
				as_long(val +
						as_double(assumed)));
	}while (assumed != old);
	return as_double(old);
}
#endif

#if (defined FLOAT && (defined(KERNEL_REDUCE) || defined(KERNEL_DOT_PROD)))
/**
 * Manual implementation of a single-precision atomic add based on an atomic compare and exchange
 * \param address The location of the initialized double precision value to accumulate to
 * \param val The value to add to that already stored in address
 * \return The old value at address that val was added to
 */
double atomicAdd_fl(__global float* address, float val)
{
	__global unsigned int * address_as_ui =
	(__global unsigned int *)address;
	//unsigned long old = *address_as_ul, assumed;
	volatile unsigned int old = atomic_add(address_as_ui, 0), assumed;
	do {
		assumed = old;
		old = atomic_cmpxchg(address_as_ui, assumed,
				as_uint(val +
						as_float(assumed)));
	}while (assumed != old);
	return as_float(old);
}
#endif

#if (defined(DOUBLE) && defined(KERNEL_DOT_PROD))
/**
 * Double precision dot-product of identically-shaped subregions of two identically-shaped 3D arrays
 * this kernel works for 3D data only.
 * \param phi1 first input array
 * \param phi2 second input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_dotProd_db(__global double *phi1, __global double *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global double * reduction, int len_, __local volatile double * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));
	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j];
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_db(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomicAdd_db(reduction,psum[0]);
}
#endif

#if (defined(FLOAT) && defined(KERNEL_DOT_PROD))
/**
 * Single precision dot-product of identically-shaped subregions of two identically-shaped 3D arrays
 * this kernel works for 3D data only.
 * \param phi1 first input array
 * \param phi2 second input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_dotProd_fl(__global float *phi1, __global float *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global float * reduction, int len_, __local volatile float * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));
	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j];
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_fl(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomicAdd_fl(reduction,psum[0]);
}
#endif

#if (defined(UNSIGNED_LONG) && defined(KERNEL_DOT_PROD))
/**
 * Unsigned long dot-product of identically-shaped subregions of two identically-shaped 3D arrays
 * this kernel works for 3D data only.
 * \param phi1 first input array
 * \param phi2 second input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_dotProd_ul(__global unsigned long *phi1, __global unsigned long *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global unsigned long * reduction, int len_, __local volatile unsigned long * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));
	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j];
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_ul(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atom_add(reduction,psum[0]);
}
#endif

#if (defined(INTEGER) && defined(KERNEL_DOT_PROD))
/**
 * Integer dot-product of identically-shaped subregions of two identically-shaped 3D arrays
 * this kernel works for 3D data only.
 * \param phi1 first input array
 * \param phi2 second input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_dotProd_in(__global int *phi1, __global int *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global int * reduction, int len_, __local volatile int * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));
	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j];
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_in(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomic_add(reduction,psum[0]);
}
#endif

#if (defined(UNSIGNED_INTEGER) && defined(KERNEL_DOT_PROD))
/**
 * Unsigned integer dot-product of identically-shaped subregions of two identically-shaped 3D arrays
 * this kernel works for 3D data only.
 * \param phi1 first input array
 * \param phi2 second input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_dotProd_ui(__global unsigned int *phi1, __global unsigned int *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global unsigned int * reduction, int len_, __local volatile unsigned int * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));
	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j];
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_ui(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomic_add(reduction,psum[0]);
}
#endif

#if (defined(DOUBLE) && defined(KERNEL_REDUCE))
/**
 * Double precision reduction sum of subregion of 3D array
 * this kernel works for 3D data only.
 * \param phi first input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_reduce_db(__global double *phi,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global double * reduction, int len_, __local volatile double * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));

	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi[x+y*i+z*i*j];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_db(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomicAdd_db(reduction,psum[0]);
}
#endif

#if (defined(FLOAT) && defined(KERNEL_REDUCE))
/**
 * Single precision reduction sum of subregion of 3D array
 * this kernel works for 3D data only.
 * \param phi first input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_reduce_fl(__global float *phi,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global float * reduction, int len_, __local volatile float * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));

	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi[x+y*i+z*i*j];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_fl(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomicAdd_fl(reduction,psum[0]);
}
#endif

#if (defined(UNSIGNED_LONG) && defined(KERNEL_REDUCE))
/**
 * Unsigned long reduction sum of subregion of 3D array
 * this kernel works for 3D data only.
 * \param phi first input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_reduce_ul(__global unsigned long *phi,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global unsigned long * reduction, int len_, __local volatile unsigned long * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));

	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi[x+y*i+z*i*j];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_ul(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atom_add(reduction,psum[0]);
}
#endif

#if (defined(INTEGER) && defined(KERNEL_REDUCE))
/**
 * Integer reduction sum of subregion of 3D array
 * this kernel works for 3D data only.
 * \param phi first input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_reduce_in(__global int *phi,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global int * reduction, int len_, __local volatile int * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));

	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi[x+y*i+z*i*j];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_in(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomic_add(reduction,psum[0]);
}
#endif

#if (defined(UNSIGNED_INTEGER) && defined(KERNEL_REDUCE))
/**
 * Unsigned int reduction sum of subregion of 3D array
 * this kernel works for 3D data only.
 * \param phi first input array
 * \param i arrays size in the X dimension
 * \param j arrays size in the Y dimension
 * \param k arrays size in the Z dimension
 * \param sx start of region to dot product in the X dimension [0,ex)
 * \param sy start of region to dot product in the Y dimension [0,ey)
 * \param sz start of region to dot product in the Z dimension [0,ez)
 * \param ex end of region to dot product in the X dimension (sx,i-1]
 * \param ey end of region to dot product in the Y dimension (sy,j-1]
 * \param ez end of region to dot product in the Z dimension (sz,k-1]
 * \param gz the number of iterations necessary to fully-compute the Z dimension in case the global work size requires more than one element per workitem
 * \param reduction the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel
 * \param len_ the length of the dynamically allocated location region, psum
 * \param psum a dynamically-allocated local memory region for computing thread and work-group partial results
 */
__kernel void kernel_reduce_ui(__global unsigned int *phi,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, __global unsigned int * reduction, int len_, __local volatile unsigned int * psum) {

	int tid, loads, x, y, z, itr;
	int boundx, boundy, boundz;
	tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1));

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	loads = gz;

	psum[tid] = 0;
	boundy = ((y >= sy) && (y <= ey));
	boundx = ((x >= sx) && (x <= ex));

	for (itr = 0; itr < loads; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z >= sz) && (z <= ez));
		if (boundx && boundy && boundz) psum[tid] += phi[x+y*i+z*i*j];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	block_reduction_ui(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomic_add(reduction,psum[0]);
}
#endif

#if (defined(DOUBLE) && defined(KERNEL_TRANSPOSE))
/**
 * Tiled transpose of a region of a double precision array
 * this kernel works for 2D data only.
 * If a subregion is utilized, it is assumed to start at zero and extend to tran_width-1 and tran_height-1
 * Any ghost elements outside the selected subregion aren't touched
 * idata and odata must not overlap, does not support in-place transpose
 * \param odata Output matrix
 * \param idata Input matrix
 * \param arr_width Width of the input matrix
 * \param arr_height Height of the input matrix
 * \param tran_width Width to transpose (<= arr_width)
 * \param tran_height Height to transpose (<= arr_height)
 * \param tile a dynamically-allocated local memory region of size (get_local_size(0)+1)*get_local_size(1)*get_local_size(2)
 */
__kernel void kernel_transpose_2d_face_db(__global double *odata, __global double *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local double * tile)
{
//    __local double tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

	int blockIdx_x, blockIdx_y;
	int gridDim_x, gridDim_y;

	// do diagonal reordering
	//The if case degenerates to the else case, no need to have both
	//if (width == height)
	//{
	//    blockIdx_y = get_group_id(0);
	//    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
	//}
	//else
	//{
	//First figure out your number among the actual grid blocks
	int bid = get_group_id(0) + get_num_groups(0)*get_group_id(1);
	//Then figure out how many logical blocks are required in each dimension
	gridDim_x = (tran_width-1+get_local_size(0))/get_local_size(0);
	gridDim_y = (tran_height-1+get_local_size(1))/get_local_size(1);
	//Then how many logical and actual grid blocks
	int logicalBlocks = gridDim_x*gridDim_y;
	int gridBlocks = get_num_groups(0)*get_num_groups(1);
	//Loop over all logical blocks
	for (; bid < logicalBlocks; bid += gridBlocks) {
		//Compute the current logical block index in each dimension
		blockIdx_y = bid%gridDim_y;
		blockIdx_x = ((bid/gridDim_y)+blockIdx_y)%gridDim_x;
		//}

		//int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
		//int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
		//int index_in = xIndex_in + (yIndex_in)*width;
		int index_in = xIndex_in + (yIndex_in)*arr_width;

		//int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
		//int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
		//int index_out = xIndex_out + (yIndex_out)*height;
		int index_out = xIndex_out + (yIndex_out)*arr_height;

		if(xIndex_in < tran_width && yIndex_in < tran_height)
		//tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
		tile[get_local_id(1)*(get_local_size(0))+get_local_id(0)] = idata[index_in];

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//if(xIndex_out < width && yIndex_out < height)
		if(xIndex_out < tran_height && yIndex_out < tran_width)
		//odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
		odata[index_out] = tile[get_local_id(1)+(get_local_size(1))*get_local_id(0)];

		//Added with the loop to ensure writes are finished before new vals go into shared memory
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
#endif

#if (defined(FLOAT) && defined(KERNEL_TRANSPOSE))
/**
 * Tiled transpose of a region of a single precision array
 * this kernel works for 2D data only.
 * If a subregion is utilized, it is assumed to start at zero and extend to tran_width-1 and tran_height-1
 * Any ghost elements outside the selected subregion aren't touched
 * idata and odata must not overlap, does not support in-place transpose
 * \param odata Output matrix
 * \param idata Input matrix
 * \param arr_width Width of the input matrix
 * \param arr_height Height of the input matrix
 * \param tran_width Width to transpose (<= arr_width)
 * \param tran_height Height to transpose (<= arr_height)
 * \param tile a dynamically-allocated local memory region of size (get_local_size(0)+1)*get_local_size(1)*get_local_size(2)
 */
__kernel void kernel_transpose_2d_face_fl(__global float *odata, __global float *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local float * tile)
{
//    __local float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

	int blockIdx_x, blockIdx_y;
	int gridDim_x, gridDim_y;

	// do diagonal reordering
	//The if case degenerates to the else case, no need to have both
	//if (width == height)
	//{
	//    blockIdx_y = get_group_id(0);
	//    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
	//}
	//else
	//{
	//First figure out your number among the actual grid blocks
	int bid = get_group_id(0) + get_num_groups(0)*get_group_id(1);
	//Then figure out how many logical blocks are required in each dimension
	gridDim_x = (tran_width-1+get_local_size(0))/get_local_size(0);
	gridDim_y = (tran_height-1+get_local_size(1))/get_local_size(1);
	//Then how many logical and actual grid blocks
	int logicalBlocks = gridDim_x*gridDim_y;
	int gridBlocks = get_num_groups(0)*get_num_groups(1);
	//Loop over all logical blocks
	for (; bid < logicalBlocks; bid += gridBlocks) {
		//Compute the current logical block index in each dimension
		blockIdx_y = bid%gridDim_y;
		blockIdx_x = ((bid/gridDim_y)+blockIdx_y)%gridDim_x;
		//}

		//int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
		//int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
		//int index_in = xIndex_in + (yIndex_in)*width;
		int index_in = xIndex_in + (yIndex_in)*arr_width;

		//int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
		//int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
		//int index_out = xIndex_out + (yIndex_out)*height;
		int index_out = xIndex_out + (yIndex_out)*arr_height;

		if(xIndex_in < tran_width && yIndex_in < tran_height)
		//tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
		tile[get_local_id(1)*(get_local_size(0))+get_local_id(0)] = idata[index_in];

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//if(xIndex_out < width && yIndex_out < height)
		if(xIndex_out < tran_height && yIndex_out < tran_width)
		//odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
		odata[index_out] = tile[get_local_id(1)+(get_local_size(1))*get_local_id(0)];

		//Added with the loop to ensure writes are finished before new vals go into shared memory
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
#endif

#if (defined(UNSIGNED_LONG) && defined(KERNEL_TRANSPOSE))
/**
 * Tiled transpose of a region of an unsigned long array
 * this kernel works for 2D data only.
 * If a subregion is utilized, it is assumed to start at zero and extend to tran_width-1 and tran_height-1
 * Any ghost elements outside the selected subregion aren't touched
 * idata and odata must not overlap, does not support in-place transpose
 * \param odata Output matrix
 * \param idata Input matrix
 * \param arr_width Width of the input matrix
 * \param arr_height Height of the input matrix
 * \param tran_width Width to transpose (<= arr_width)
 * \param tran_height Height to transpose (<= arr_height)
 * \param tile a dynamically-allocated local memory region of size (get_local_size(0)+1)*get_local_size(1)*get_local_size(2)
 */
__kernel void kernel_transpose_2d_face_ul(__global unsigned long *odata, __global unsigned long *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local unsigned long * tile)
{
//    __local unsigned long tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

	int blockIdx_x, blockIdx_y;
	int gridDim_x, gridDim_y;

	// do diagonal reordering
	//The if case degenerates to the else case, no need to have both
	//if (width == height)
	//{
	//    blockIdx_y = get_group_id(0);
	//    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
	//}
	//else
	//{
	//First figure out your number among the actual grid blocks
	int bid = get_group_id(0) + get_num_groups(0)*get_group_id(1);
	//Then figure out how many logical blocks are required in each dimension
	gridDim_x = (tran_width-1+get_local_size(0))/get_local_size(0);
	gridDim_y = (tran_height-1+get_local_size(1))/get_local_size(1);
	//Then how many logical and actual grid blocks
	int logicalBlocks = gridDim_x*gridDim_y;
	int gridBlocks = get_num_groups(0)*get_num_groups(1);
	//Loop over all logical blocks
	for (; bid < logicalBlocks; bid += gridBlocks) {
		//Compute the current logical block index in each dimension
		blockIdx_y = bid%gridDim_y;
		blockIdx_x = ((bid/gridDim_y)+blockIdx_y)%gridDim_x;
		//}

		//int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
		//int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
		//int index_in = xIndex_in + (yIndex_in)*width;
		int index_in = xIndex_in + (yIndex_in)*arr_width;

		//int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
		//int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
		//int index_out = xIndex_out + (yIndex_out)*height;
		int index_out = xIndex_out + (yIndex_out)*arr_height;

		if(xIndex_in < tran_width && yIndex_in < tran_height)
		//tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
		tile[get_local_id(1)*(get_local_size(0))+get_local_id(0)] = idata[index_in];

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//if(xIndex_out < width && yIndex_out < height)
		if(xIndex_out < tran_height && yIndex_out < tran_width)
		//odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
		odata[index_out] = tile[get_local_id(1)+(get_local_size(1))*get_local_id(0)];

		//Added with the loop to ensure writes are finished before new vals go into shared memory
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
#endif

#if (defined(INTEGER) && defined(KERNEL_TRANSPOSE))
/**
 * Tiled transpose of a region of an integer array
 * this kernel works for 2D data only.
 * If a subregion is utilized, it is assumed to start at zero and extend to tran_width-1 and tran_height-1
 * Any ghost elements outside the selected subregion aren't touched
 * idata and odata must not overlap, does not support in-place transpose
 * \param odata Output matrix
 * \param idata Input matrix
 * \param arr_width Width of the input matrix
 * \param arr_height Height of the input matrix
 * \param tran_width Width to transpose (<= arr_width)
 * \param tran_height Height to transpose (<= arr_height)
 * \param tile a dynamically-allocated local memory region of size (get_local_size(0)+1)*get_local_size(1)*get_local_size(2)
 */
__kernel void kernel_transpose_2d_face_in(__global int *odata, __global int *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local int * tile)
{
//    __local int tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

	int blockIdx_x, blockIdx_y;
	int gridDim_x, gridDim_y;

	// do diagonal reordering
	//The if case degenerates to the else case, no need to have both
	//if (width == height)
	//{
	//    blockIdx_y = get_group_id(0);
	//    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
	//}
	//else
	//{
	//First figure out your number among the actual grid blocks
	int bid = get_group_id(0) + get_num_groups(0)*get_group_id(1);
	//Then figure out how many logical blocks are required in each dimension
	gridDim_x = (tran_width-1+get_local_size(0))/get_local_size(0);
	gridDim_y = (tran_height-1+get_local_size(1))/get_local_size(1);
	//Then how many logical and actual grid blocks
	int logicalBlocks = gridDim_x*gridDim_y;
	int gridBlocks = get_num_groups(0)*get_num_groups(1);
	//Loop over all logical blocks
	for (; bid < logicalBlocks; bid += gridBlocks) {
		//Compute the current logical block index in each dimension
		blockIdx_y = bid%gridDim_y;
		blockIdx_x = ((bid/gridDim_y)+blockIdx_y)%gridDim_x;
		//}

		//int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
		//int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
		//int index_in = xIndex_in + (yIndex_in)*width;
		int index_in = xIndex_in + (yIndex_in)*arr_width;

		//int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
		//int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
		//int index_out = xIndex_out + (yIndex_out)*height;
		int index_out = xIndex_out + (yIndex_out)*arr_height;

		if(xIndex_in < tran_width && yIndex_in < tran_height)
		//tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
		tile[get_local_id(1)*(get_local_size(0))+get_local_id(0)] = idata[index_in];

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//if(xIndex_out < width && yIndex_out < height)
		if(xIndex_out < tran_height && yIndex_out < tran_width)
		//odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
		odata[index_out] = tile[get_local_id(1)+(get_local_size(1))*get_local_id(0)];

		//Added with the loop to ensure writes are finished before new vals go into shared memory
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
#endif

#if (defined(UNSIGNED_INTEGER) && defined(KERNEL_TRANSPOSE))
/**
 * Tiled transpose of a region of an unsigned integer array
 * this kernel works for 2D data only.
 * If a subregion is utilized, it is assumed to start at zero and extend to tran_width-1 and tran_height-1
 * Any ghost elements outside the selected subregion aren't touched
 * idata and odata must not overlap, does not support in-place transpose
 * \param odata Output matrix
 * \param idata Input matrix
 * \param arr_width Width of the input matrix
 * \param arr_height Height of the input matrix
 * \param tran_width Width to transpose (<= arr_width)
 * \param tran_height Height to transpose (<= arr_height)
 * \param tile a dynamically-allocated local memory region of size (get_local_size(0)+1)*get_local_size(1)*get_local_size(2)
 */
__kernel void kernel_transpose_2d_face_ui(__global unsigned int *odata, __global unsigned int *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local unsigned int * tile)
{
//    __local unsigned int tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

	int blockIdx_x, blockIdx_y;
	int gridDim_x, gridDim_y;

	// do diagonal reordering
	//The if case degenerates to the else case, no need to have both
	//if (width == height)
	//{
	//    blockIdx_y = get_group_id(0);
	//    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
	//}
	//else
	//{
	//First figure out your number among the actual grid blocks
	int bid = get_group_id(0) + get_num_groups(0)*get_group_id(1);
	//Then figure out how many logical blocks are required in each dimension
	gridDim_x = (tran_width-1+get_local_size(0))/get_local_size(0);
	gridDim_y = (tran_height-1+get_local_size(1))/get_local_size(1);
	//Then how many logical and actual grid blocks
	int logicalBlocks = gridDim_x*gridDim_y;
	int gridBlocks = get_num_groups(0)*get_num_groups(1);
	//Loop over all logical blocks
	for (; bid < logicalBlocks; bid += gridBlocks) {
		//Compute the current logical block index in each dimension
		blockIdx_y = bid%gridDim_y;
		blockIdx_x = ((bid/gridDim_y)+blockIdx_y)%gridDim_x;
		//}

		//int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
		//int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
		//int index_in = xIndex_in + (yIndex_in)*width;
		int index_in = xIndex_in + (yIndex_in)*arr_width;

		//int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
		int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
		//int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
		int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
		//int index_out = xIndex_out + (yIndex_out)*height;
		int index_out = xIndex_out + (yIndex_out)*arr_height;

		if(xIndex_in < tran_width && yIndex_in < tran_height)
		//tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
		tile[get_local_id(1)*(get_local_size(0))+get_local_id(0)] = idata[index_in];

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//if(xIndex_out < width && yIndex_out < height)
		if(xIndex_out < tran_height && yIndex_out < tran_width)
		//odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
		odata[index_out] = tile[get_local_id(1)+(get_local_size(1))*get_local_id(0)];

		//Added with the loop to ensure writes are finished before new vals go into shared memory
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
#endif

#if (defined(KERNEL_PACK) || defined(KERNEL_UNPACK))
/**
 * Internal function to translate a thread ID to the corresponding linear index into a 3D array, based on a specified face, used by the pack and unpack kernels.
 * \param tid The index in the packed linear buffer
 * \param a A dynamically-allocated local memory array used to calculate partial offsets during the level iteration
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \return the linear index in the unpacked 3D array 
 */
int get_pack_index (int tid, __local int * a, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size) {
	int i, j, k, l;
	int pos;
	for(i = 0; i < count; i++)
	a[tid%get_local_size(0) + i * get_local_size(0)] = 0;

	for(i = 0; i < count; i++)
	{
		k = 0;
		for(j = 0; j < i; j++)
		{
			k += a[tid%get_local_size(0) + j * get_local_size(0)] * c_face_child_size[j];
		}
		l = c_face_child_size[i];
		for(j = 0; j < c_face_size[i]; j++)
		{
			if (tid - k < l)
			break;
			else
			l += c_face_child_size[i];
		}
		a[tid%get_local_size(0) + i * get_local_size(0)] = j;
	}
	pos = start;
	for(i = 0; i < count; i++)
	{
		pos += a[tid%get_local_size(0) + i * get_local_size(0)] * c_face_stride[i];
	}
	return pos;
}
#endif

#if (defined(DOUBLE) && defined(KERNEL_PACK))
/**
 * Kernel to pack a specified slab of a 3D buffer into a packed format for communication over the network.
 * Can be used with a unit-thick Slab to produce a 2D array for transposition
 * \param packed_buf The output buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_pack_2d_face_db(__global double *packed_buf, __global double *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}
#endif

#if (defined(FLOAT) && defined (KERNEL_PACK))
/**
 * Kernel to pack a specified slab of a 3D buffer into a packed format for communication over the network.
 * Can be used with a unit-thick Slab to produce a 2D array for transposition
 * \param packed_buf The output buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_pack_2d_face_fl(__global float *packed_buf, __global float *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}
#endif

#if (defined(UNSIGNED_LONG) && defined(KERNEL_PACK))
/**
 * Kernel to pack a specified slab of a 3D buffer into a packed format for communication over the network.
 * Can be used with a unit-thick Slab to produce a 2D array for transposition
 * \param packed_buf The output buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_pack_2d_face_ul(__global unsigned long *packed_buf, __global unsigned long *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}
#endif

#if (defined(INTEGER) && defined(KERNEL_PACK))
/**
 * Kernel to pack a specified slab of a 3D buffer into a packed format for communication over the network.
 * Can be used with a unit-thick Slab to produce a 2D array for transposition
 * \param packed_buf The output buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_pack_2d_face_in(__global int *packed_buf, __global int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}
#endif

#if (defined(UNSIGNED_INTEGER) && defined(KERNEL_PACK))
/**
 * Kernel to pack a specified slab of a 3D buffer into a packed format for communication over the network.
 * Can be used with a unit-thick Slab to produce a 2D array for transposition
 * \param packed_buf The output buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_pack_2d_face_ui(__global unsigned int *packed_buf, __global unsigned int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}
#endif

#if (defined(DOUBLE) && defined(KERNEL_UNPACK))
/**
 * Kernel to unpack a specified slab of a 3D buffer from a packed format likely communicated over the network.
 * Can be used with a unit-thick slab to populate a portion of a 3D region from a 2D array
 * \param packed_buf The inpu buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_unpack_2d_face_db(__global double *packed_buf, __global double *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}
#endif

#if (defined(FLOAT) && defined(KERNEL_UNPACK))
/**
 * Kernel to unpack a specified slab of a 3D buffer from a packed format likely communicated over the network.
 * Can be used with a unit-thick slab to populate a portion of a 3D region from a 2D array
 * \param packed_buf The inpu buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_unpack_2d_face_fl(__global float *packed_buf, __global float *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}
#endif

#if (defined(UNSIGNED_LONG) && defined(KERNEL_UNPACK))
/**
 * Kernel to unpack a specified slab of a 3D buffer from a packed format likely communicated over the network.
 * Can be used with a unit-thick slab to populate a portion of a 3D region from a 2D array
 * \param packed_buf The inpu buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_unpack_2d_face_ul(__global unsigned long *packed_buf, __global unsigned long *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}
#endif

#if (defined(INTEGER) && defined(KERNEL_UNPACK))
/**
 * Kernel to unpack a specified slab of a 3D buffer from a packed format likely communicated over the network.
 * Can be used with a unit-thick slab to populate a portion of a 3D region from a 2D array
 * \param packed_buf The inpu buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_unpack_2d_face_in(__global int *packed_buf, __global int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}
#endif

#if (defined(UNSIGNED_INTEGER) && defined(KERNEL_UNPACK))
/**
 * Kernel to unpack a specified slab of a 3D buffer from a packed format likely communicated over the network.
 * Can be used with a unit-thick slab to populate a portion of a 3D region from a 2D array
 * \param packed_buf The inpu buffer, of sufficient size to store the entire slab
 * \param buf The unpacked 3D buffer
 * \param size The length of packed_buf
 * \param start The initial index offset
 * \param count The number of elements in the size, stride, and child_size arrays 
 * \param c_face_size The number of samples in each tree level (dimension)
 * \param c_face_stride The linear distance between samples in each tree level
 * \param c_face_child_size The number of samples in the next tree level and all its descendants
 * \param a A dynamically-allocated local memory array used to during pack offset calculation
 */
__kernel void kernel_unpack_2d_face_ui(__global unsigned int *packed_buf, __global unsigned int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
	int idx = get_global_id(0);
	const int nthreads = get_global_size(0);
	// this loop handles both nthreads > size and nthreads < size
	for (; idx < size; idx += nthreads)
	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}
#endif

/**
 * Kernel to perform a single step of a 3D 7-point Jacobi stencil, averaging the value of the 7 cells into the center
 * this kernel works for 3D data only, assumes a minimum of 1-element halo region exists on all sides
 * //Read-only cache + Register blocking (Z)
 * works only with 2D thread blocks
 * \param ind Input array
 * \param outd Output array
 * \param i X dimension of ind and outd
 * \param j Y dimension of ind and outd
 * \param k Z dimension of ind and outd
 * \param sx Start index in the X dimension [1,ex)
 * \param sy Start index in the Y dimension [1,ey)
 * \param sz Start index in the Z dimension [1,ez)
 * \param ex End index in the X dimension (sx,i-1]
 * \param ey End index in the Y dimension (sy,j-1]
 * \param ez End index in the Z dimension (sz,k-1]
 * \param gz The number of iterations necessary to cover the entire Z dimesion if the global work size can't do it in one
 * \param len_ the length of the bind array (only used by V2)
 * \param bind A dynamically allocated local memory array used for local memory blocking (only used by V2) 
 */
#if (defined(DOUBLE) && defined(KERNEL_STENCIL))
__kernel void kernel_stencil_3d7p_db(const __global double * __restrict__ ind, __global double * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_, __local double * bind) {
	double r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;
	z = get_local_id(2) +sz; //blockDim.z ==1
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
				ind[c+i] + ind[c+1] + rz2 ) / (double) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
	}
}
#endif

//FIXME PS2018: why is this disabled?
#if (0 && defined(DOUBLE) && defined(KERNEL_STENCIL_V2))
// work with 2D and 3D thread blocks
__kernel void kernel_stencil_3d7p_db_v0(const __global double * __restrict__ ind, __global double * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_, __local double * bind) {
	int x, y, z, itr;
	bool boundx, boundy, boundz;

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;

	boundy = ((y > sy) && (y < ey));
	boundx = ((x > sx) && (x < ex));

	for (itr = 0; itr < gz; itr++) {
		z = itr*get_local_size(2)+get_local_id(2) +sz;
		boundz = ((z > sz) && (z < ez));
		if (boundx && boundy && boundz)
		outd[x+y*i+z*i*j] = ( ind[x+y*i+(z-1)*i*j] + ind[(x-1)+y*i+z*i*j] + ind[x+(y-1)*i+z*i*j] +
				ind[x+y*i+z*i*j] + ind[x+(y+1)*i+z*i*j] + ind[(x+1)+y*i+z*i*j] +
				ind[x+y*i+(z+1)*i*j] ) / (double) 7;
	}
}
#endif

//Read-only cache + Rigster blocking (Z) + smem blocking (X-Y)
// work only with 2D thread blocks (use rectangular blocks, i.e. 64*4, 128*2)
#if (0 && defined(DOUBLE) && defined(KERNEL_STENCIL_V2))
__kernel void kernel_stencil_3d7p_db_v2(const __global double * __restrict__ ind, __global double * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_, __local double * bind) {
	const int bi = (get_local_size(0)+2);
	const int bc = (get_local_id(0)+1)+(get_local_id(1)+1)*bi;
	double r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;
	z = get_local_id(2) +sz; //blockDim.z ==1
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

		if(get_local_id(0) == 0)
		bind[bc-1] = ind[c-1];
		else if (get_local_id(0) == get_local_size(0)-1)
		bind[bc+1] = ind[c+1];

		if(get_local_id(1) == 0)
		bind[bc-bi] = ind[c-i];
		else if (get_local_id(1) == get_local_size(1)-1)
		bind[bc+bi] = ind[c+i];

		//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		barrier(CLK_LOCAL_MEM_FENCE);

		if (boundx && boundy && boundz)
		outd[c] = ( rz1 + bind[bc-1] + bind[bc-bi] + r0 +
				bind[bc+bi] + bind[bc+1] + rz2 ) / (double) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
#endif

#if (defined(FLOAT) && defined(KERNEL_STENCIL))
/**
 * Kernel to perform a single step of a 3D 7-point Jacobi stencil, averaging the value of the 7 cells into the center
 * this kernel works for 3D data only, assumes a minimum of 1-element halo region exists on all sides
 * //Read-only cache + Register blocking (Z)
 * works only with 2D thread blocks
 * \param ind Input array
 * \param outd Output array
 * \param i X dimension of ind and outd
 * \param j Y dimension of ind and outd
 * \param k Z dimension of ind and outd
 * \param sx Start index in the X dimension [1,ex)
 * \param sy Start index in the Y dimension [1,ey)
 * \param sz Start index in the Z dimension [1,ez)
 * \param ex End index in the X dimension (sx,i-1]
 * \param ey End index in the Y dimension (sy,j-1]
 * \param ez End index in the Z dimension (sz,k-1]
 * \param gz The number of iterations necessary to cover the entire Z dimesion if the global work size can't do it in one
 * \param len_ the length of the bind array (only used by V2)
 * \param bind A dynamically allocated local memory array used for local memory blocking (only used by V2) 
 */
__kernel void kernel_stencil_3d7p_fl(const __global float * __restrict__ ind, __global float * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_, __local float * bind) {
	float r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;
	z = get_local_id(2) +sz; //blockDim.z ==1
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
				ind[c+i] + ind[c+1] + rz2 ) / (float) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
	}
}
#endif

#if (defined(UNSIGNED_LONG) && defined(KERNEL_STENCIL))
/**
 * Kernel to perform a single step of a 3D 7-point Jacobi stencil, averaging the value of the 7 cells into the center
 * this kernel works for 3D data only, assumes a minimum of 1-element halo region exists on all sides
 * //Read-only cache + Register blocking (Z)
 * works only with 2D thread blocks
 * \param ind Input array
 * \param outd Output array
 * \param i X dimension of ind and outd
 * \param j Y dimension of ind and outd
 * \param k Z dimension of ind and outd
 * \param sx Start index in the X dimension [1,ex)
 * \param sy Start index in the Y dimension [1,ey)
 * \param sz Start index in the Z dimension [1,ez)
 * \param ex End index in the X dimension (sx,i-1]
 * \param ey End index in the Y dimension (sy,j-1]
 * \param ez End index in the Z dimension (sz,k-1]
 * \param gz The number of iterations necessary to cover the entire Z dimesion if the global work size can't do it in one
 * \param len_ the length of the bind array (only used by V2)
 * \param bind A dynamically allocated local memory array used for local memory blocking (only used by V2) 
 */
__kernel void kernel_stencil_3d7p_ul(const __global unsigned long * __restrict__ ind, __global unsigned long * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_, __local unsigned long * bind) {
	unsigned long r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;
	z = get_local_id(2) +sz; //blockDim.z ==1
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
				ind[c+i] + ind[c+1] + rz2 ) / (unsigned long) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
	}
}
#endif

#if (defined(INTEGER) && defined(KERNEL_STENCIL))
/**
 * Kernel to perform a single step of a 3D 7-point Jacobi stencil, averaging the value of the 7 cells into the center
 * this kernel works for 3D data only, assumes a minimum of 1-element halo region exists on all sides
 * //Read-only cache + Register blocking (Z)
 * works only with 2D thread blocks
 * \param ind Input array
 * \param outd Output array
 * \param i X dimension of ind and outd
 * \param j Y dimension of ind and outd
 * \param k Z dimension of ind and outd
 * \param sx Start index in the X dimension [1,ex)
 * \param sy Start index in the Y dimension [1,ey)
 * \param sz Start index in the Z dimension [1,ez)
 * \param ex End index in the X dimension (sx,i-1]
 * \param ey End index in the Y dimension (sy,j-1]
 * \param ez End index in the Z dimension (sz,k-1]
 * \param gz The number of iterations necessary to cover the entire Z dimesion if the global work size can't do it in one
 * \param len_ the length of the bind array (only used by V2)
 * \param bind A dynamically allocated local memory array used for local memory blocking (only used by V2) 
 */
__kernel void kernel_stencil_3d7p_in(const __global int * __restrict__ ind, __global int * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_, __local int * bind) {
	int r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;
	z = get_local_id(2) +sz; //blockDim.z ==1
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
				ind[c+i] + ind[c+1] + rz2 ) / (int) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
	}
}
#endif


#if (defined(UNSIGNED_INTEGER) && defined(KERNEL_STENCIL))
/**
 * Kernel to perform a single step of a 3D 7-point Jacobi stencil, averaging the value of the 7 cells into the center
 * this kernel works for 3D data only, assumes a minimum of 1-element halo region exists on all sides
 * //Read-only cache + Register blocking (Z)
 * works only with 2D thread blocks
 * \param ind Input array
 * \param outd Output array
 * \param i X dimension of ind and outd
 * \param j Y dimension of ind and outd
 * \param k Z dimension of ind and outd
 * \param sx Start index in the X dimension [1,ex)
 * \param sy Start index in the Y dimension [1,ey)
 * \param sz Start index in the Z dimension [1,ez)
 * \param ex End index in the X dimension (sx,i-1]
 * \param ey End index in the Y dimension (sy,j-1]
 * \param ez End index in the Z dimension (sz,k-1]
 * \param gz The number of iterations necessary to cover the entire Z dimesion if the global work size can't do it in one
 * \param len_ the length of the bind array (only used by V2)
 * \param bind A dynamically allocated local memory array used for local memory blocking (only used by V2) 
 */
__kernel void kernel_stencil_3d7p_ui(const __global unsigned int * __restrict__ ind, __global unsigned int * __restrict__ outd,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez,
		int gz, int len_, __local unsigned int * bind) {
	unsigned int r0, rz1, rz2;
	int x, y, z;
	int ij = i*j;
	int c;
	bool boundx, boundy, boundz;

	x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx;
	y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy;
	z = get_local_id(2) +sz; //blockDim.z ==1
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
				ind[c+i] + ind[c+1] + rz2 ) / (unsigned int) 7;
		c += ij;
		rz1 = r0;
		r0 = rz2;
		rz2 = ind[c+ij];
	}
}
#endif


#if (defined(DOUBLE) && defined (KERNEL_CSR))
/**
 * Kernel to compute an SPMV where the matrix is stored in CSR format
 * Only works with 1D worksizes
 * \param num_rows The number of rows in the CSR matrix
 * \param Ap The row start offset array of matrix A
 * \param Aj The column index array of matrix A
 * \param Ax The data values of matrix A
 * \param x the input vector to multiply by
 * \param y the output vector
 */
__kernel void kernel_csr_db(const unsigned int num_rows,
                __global unsigned int * Ap,
                __global unsigned int * Aj,
                __global double * Ax,
                __global double * x,
                __global double * y) {
	
	unsigned int row = get_global_id(0);

	if(row < num_rows)
	{
		double sum = y[row];

		const unsigned int row_start = Ap[row];
		const unsigned int row_end = Ap[row+1];

		unsigned int jj = 0;
		for (jj = row_start; jj < row_end; jj++)
				sum += Ax[jj] * x[Aj[jj]];

		y[row] = sum;
	}
}
#endif


#if (defined(FLOAT) && defined (KERNEL_CSR))
/**
 * Kernel to compute an SPMV where the matrix is stored in CSR format
 * Only works with 1D worksizes
 * \param num_rows The number of rows in the CSR matrix
 * \param Ap The row start offset array of matrix A
 * \param Aj The column index array of matrix A
 * \param Ax The data values of matrix A
 * \param x the input vector to multiply by
 * \param y the output vector
 */
__kernel void kernel_csr_fl(const unsigned int num_rows,
                __global unsigned int * Ap,
                __global unsigned int * Aj,
                __global float * Ax,
                __global float * x,
                __global float * y) {
	
	unsigned int row = get_global_id(0);

	if(row < num_rows)
	{
		float sum = y[row];

		const unsigned int row_start = Ap[row];
		const unsigned int row_end = Ap[row+1];

		unsigned int jj = 0;
		for (jj = row_start; jj < row_end; jj++)
				sum += Ax[jj] * x[Aj[jj]];

		y[row] = sum;
	}
}
#endif


#if (defined(UNSIGNED_LONG) && defined (KERNEL_CSR))
/**
 * Kernel to compute an SPMV where the matrix is stored in CSR format
 * Only works with 1D worksizes
 * \param num_rows The number of rows in the CSR matrix
 * \param Ap The row start offset array of matrix A
 * \param Aj The column index array of matrix A
 * \param Ax The data values of matrix A
 * \param x the input vector to multiply by
 * \param y the output vector
 */
__kernel void kernel_csr_ul(const unsigned int num_rows,
                __global unsigned int * Ap,
                __global unsigned int * Aj,
                __global unsigned long * Ax,
                __global unsigned long * x,
                __global unsigned long * y) {
	
	unsigned int row = get_global_id(0);

	if(row < num_rows)
	{
		unsigned long sum = y[row];

		const unsigned int row_start = Ap[row];
		const unsigned int row_end = Ap[row+1];

		unsigned int jj = 0;
		for (jj = row_start; jj < row_end; jj++)
				sum += Ax[jj] * x[Aj[jj]];

		y[row] = sum;
	}
}
#endif


#if (defined(INTEGER) && defined (KERNEL_CSR))
/**
 * Kernel to compute an SPMV where the matrix is stored in CSR format
 * Only works with 1D worksizes
 * \param num_rows The number of rows in the CSR matrix
 * \param Ap The row start offset array of matrix A
 * \param Aj The column index array of matrix A
 * \param Ax The data values of matrix A
 * \param x the input vector to multiply by
 * \param y the output vector
 */
__kernel void kernel_csr_in(const unsigned int num_rows,
                __global unsigned int * Ap,
                __global unsigned int * Aj,
                __global int * Ax,
                __global int * x,
                __global int * y) {
	
	unsigned int row = get_global_id(0);

	if(row < num_rows)
	{
		int sum = y[row];

		const unsigned int row_start = Ap[row];
		const unsigned int row_end = Ap[row+1];

		unsigned int jj = 0;
		for (jj = row_start; jj < row_end; jj++)
				sum += Ax[jj] * x[Aj[jj]];

		y[row] = sum;
	}
}
#endif


#if (defined(UNSIGNED_INTEGER) && defined (KERNEL_CSR))
/**
 * Kernel to compute an SPMV where the matrix is stored in CSR format
 * Only works with 1D worksizes
 * \param num_rows The number of rows in the CSR matrix
 * \param Ap The row start offset array of matrix A
 * \param Aj The column index array of matrix A
 * \param Ax The data values of matrix A
 * \param x the input vector to multiply by
 * \param y the output vector
 */
__kernel void kernel_csr_ui(const unsigned int num_rows,
                __global unsigned int * Ap,
                __global unsigned int * Aj,
                __global unsigned int * Ax,
                __global unsigned int * x,
                __global unsigned int * y) {
	
	unsigned int row = get_global_id(0);

	if(row < num_rows)
	{
		unsigned int sum = y[row];

		const unsigned int row_start = Ap[row];
		const unsigned int row_end = Ap[row+1];

		unsigned int jj = 0;
		for (jj = row_start; jj < row_end; jj++)
				sum += Ax[jj] * x[Aj[jj]];

		y[row] = sum;
	}
}
#endif

#ifdef KERNEL_CRC
//TODO either include this in the repo or make it required and found before build
#define OPENCL
#include "metamorph-backends/opencl-backend/eth_crc32_lut.h"
#endif

//CRC kernel 
#if ((defined(DOUBLE) || defined(FLOAT) || defined(UNSIGNED_LONG) || defined(INTEGER) || defined(UNSIGNED_INTEGER)) && defined(KERNEL_CRC))
/**
 * Kernel to compute a cyclic redundancy check on a stretch of binary data
 * Only supports 1D worksizes and has no concept of the main MetaMorph data types
 * Uses the canonical Slice-by-8 algorithm
 * \todo FIXME This looks like a single-work-item variant, need to validate and ensure an NDRange version also exists
 * \param data The data buffer to evaluate
 * \param length_bytes Number of bytes for each thread to process
 * \param length_ints Number of integers for each thread to process (should be length_bytes/sizeof(uint))
 * \param num_pages How many pages each thread iterates over
 * \param res The result buffer
 */
__kernel void kernel_crc_ui(__global const uint* restrict data, 
		uint length_bytes, 
		const uint length_ints,
		uint num_pages ,
		__global uint* restrict res)
{

	__private uint crc;
	__private uchar* currentChar;
	__private uint one,two;
	__private size_t i,j,gid;
	 uint pages = num_pages;
        uint loc_length_bytes = length_bytes;

	crc = 0xFFFFFFFF;
	gid = 0;

for(gid = 0; gid <  pages ; gid++)
{
	i = gid * length_ints;
	loc_length_bytes = length_bytes;
        crc = 0xFFFFFFFF;
	while (loc_length_bytes >= 8) // process eight bytes at once
	{
		one = data[i++] ^ crc;
		two = data[i++];
		crc = crc32Lookup[7][ one      & 0xFF] ^
			crc32Lookup[6][(one>> 8) & 0xFF] ^
			crc32Lookup[5][(one>>16) & 0xFF] ^
			crc32Lookup[4][ one>>24        ] ^
			crc32Lookup[3][ two      & 0xFF] ^
			crc32Lookup[2][(two>> 8) & 0xFF] ^
			crc32Lookup[1][(two>>16) & 0xFF] ^
			crc32Lookup[0][ two>>24        ];
		loc_length_bytes -= 8;
	}

	while(loc_length_bytes) // remaining 1 to 7 bytes
	{
		one = data[i++];
		currentChar = (unsigned char*) &one;
		j=0;
		while (loc_length_bytes && j < 4) 
		{
			loc_length_bytes = loc_length_bytes - 1;
			crc = (crc >> 8) ^ crc32Lookup[0][(crc & 0xFF) ^ currentChar[j]];
			j = j + 1;
		}
	}

	res[gid] = ~crc;
}
}
#endif 
