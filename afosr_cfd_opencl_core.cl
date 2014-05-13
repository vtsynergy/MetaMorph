#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable


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

//Paul - Implementation of double atomicAdd from CUDA Programming Guide: Appendix B.12
// ported to OpenCL
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
	} while (assumed != old);
	return as_double(old);
}

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
	} while (assumed != old);
	return as_float(old);
}

//  !this kernel works for 3D data only.
//  ! PHI1 and PHI2 are input arrays.
//  ! s* parameters are start values in each dimension.
//  ! e* parameters are end values in each dimension.
//  ! s* and e* are only necessary when the halo layers 
//  !   has different thickness along various directions.
//  ! i,j,k are the array dimensions
//  ! len_ is number of threads in a threadblock.
//  !      This can be computed in the kernel itself.
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


__kernel void kernel_reduce(__global unsigned int *phi,
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



