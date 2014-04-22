#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable


void block_reduction(__local volatile double *psum, int tid, int len_) {
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
double atomicAdd(__global double* address, double val)
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

//  !this kernel works for 3D data only.
//  ! PHI1 and PHI2 are input arrays.
//  ! s* parameters are start values in each dimension.
//  ! e* parameters are end values in each dimension.
//  ! s* and e* are only necessary when the halo layers 
//  !   has different thickness along various directions.
//  ! i,j,k are the array dimensions
//  ! len_ is number of threads in a threadblock.
//  !      This can be computed in the kernel itself.
__kernel void kernel_dotProd(__global double *phi1, __global double *phi2,
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
	block_reduction(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(tid == 0) atomicAdd(reduction,psum[0]);
}


__kernel void kernel_reduction3(__global double *phi,
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
	block_reduction(psum,tid,len_);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	if(tid == 0) atomicAdd(reduction,psum[0]);
}



