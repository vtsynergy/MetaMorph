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

__kernel void kernel_transpose_2d_db(__global double *odata, __global double *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local double * tile)
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
        tile[get_local_id(1)*(get_local_size(0)+1)+get_local_id(0)] =  idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //if(xIndex_out < width && yIndex_out < height)
    if(xIndex_out < tran_height && yIndex_out < tran_width)
        //odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
        odata[index_out] = tile[get_local_id(0)+(get_local_size(0)+1)*get_local_id(1)];

    //Added with the loop to ensure writes are finished before new vals go into shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void kernel_transpose_2d_fl(__global float *odata, __global float *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local float * tile)
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
        tile[get_local_id(1)*(get_local_size(0)+1)+get_local_id(0)] =  idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //if(xIndex_out < width && yIndex_out < height)
    if(xIndex_out < tran_height && yIndex_out < tran_width)
        //odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
        odata[index_out] = tile[get_local_id(0)+(get_local_size(0)+1)*get_local_id(1)];

    //Added with the loop to ensure writes are finished before new vals go into shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void kernel_transpose_2d_ul(__global unsigned long *odata, __global unsigned long *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local unsigned long * tile)
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
        tile[get_local_id(1)*(get_local_size(0)+1)+get_local_id(0)] =  idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //if(xIndex_out < width && yIndex_out < height)
    if(xIndex_out < tran_height && yIndex_out < tran_width)
        //odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
        odata[index_out] = tile[get_local_id(0)+(get_local_size(0)+1)*get_local_id(1)];

    //Added with the loop to ensure writes are finished before new vals go into shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void kernel_transpose_2d_in(__global int *odata, __global int *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local int * tile)
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
        tile[get_local_id(1)*(get_local_size(0)+1)+get_local_id(0)] =  idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //if(xIndex_out < width && yIndex_out < height)
    if(xIndex_out < tran_height && yIndex_out < tran_width)
        //odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
        odata[index_out] = tile[get_local_id(0)+(get_local_size(0)+1)*get_local_id(1)];

    //Added with the loop to ensure writes are finished before new vals go into shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void kernel_transpose_2d_ui(__global unsigned int *odata, __global unsigned int *idata, int arr_width, int arr_height, int tran_width, int tran_height, __local unsigned int * tile)
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
        tile[get_local_id(1)*(get_local_size(0)+1)+get_local_id(0)] =  idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //if(xIndex_out < width && yIndex_out < height)
    if(xIndex_out < tran_height && yIndex_out < tran_width)
        //odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
        odata[index_out] = tile[get_local_id(0)+(get_local_size(0)+1)*get_local_id(1)];

    //Added with the loop to ensure writes are finished before new vals go into shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}
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

__kernel void kernel_pack_db(__global double *packed_buf, __global double *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_fl(__global float *packed_buf, __global float *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_ul(__global unsigned long *packed_buf, __global unsigned long *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_in(__global int *packed_buf, __global int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_ui(__global unsigned int *packed_buf, __global unsigned int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
	packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)];
}



__kernel void kernel_unpack_db(__global double *packed_buf, __global double *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
    	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_fl(__global float *packed_buf, __global float *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
    	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_ul(__global unsigned long *packed_buf, __global unsigned long *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
    	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_in(__global int *packed_buf, __global int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
    	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_ui(__global unsigned int *packed_buf, __global unsigned int *buf, int size, int start, int count, __constant int * c_face_size, __constant int * c_face_stride, __constant int * c_face_child_size, __local int *a)
{
    int idx = get_global_id(0);
    const int nthreads = get_global_size(0);
    // this loop handles both nthreads > size and nthreads < size
    for (; idx < size; idx += nthreads)
    	buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride, c_face_child_size)] = packed_buf[idx];
}

