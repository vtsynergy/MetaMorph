/** OpenCL Back-End: MIC customization **/

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

void block_reduction_db(__local volatile double *psum, int tid, int len_) {
  int stride = len_ >> 1;
  while (stride > 0) {
    if (tid < stride)
      psum[tid] += psum[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    stride >>= 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop
  // from (stride >32) to ensure compatibility with CPU platforms.
  // once I can work out how to use the preferred_work_group_size_multiple that
  // Tom suggested, I'll re-add the optimized version.
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
    if (tid < stride)
      psum[tid] += psum[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    stride >>= 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop
  // from (stride >32) to ensure compatibility with CPU platforms.
  // once I can work out how to use the preferred_work_group_size_multiple that
  // Tom suggested, I'll re-add the optimized version.
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

void block_reduction_ul(__local volatile unsigned long *psum, int tid,
                        int len_) {
  int stride = len_ >> 1;
  while (stride > 0) {
    if (tid < stride)
      psum[tid] += psum[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    stride >>= 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop
  // from (stride >32) to ensure compatibility with CPU platforms.
  // once I can work out how to use the preferred_work_group_size_multiple that
  // Tom suggested, I'll re-add the optimized version.
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
    if (tid < stride)
      psum[tid] += psum[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    stride >>= 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop
  // from (stride >32) to ensure compatibility with CPU platforms.
  // once I can work out how to use the preferred_work_group_size_multiple that
  // Tom suggested, I'll re-add the optimized version.
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

void block_reduction_ui(__local volatile unsigned int *psum, int tid,
                        int len_) {
  int stride = len_ >> 1;
  while (stride > 0) {
    if (tid < stride)
      psum[tid] += psum[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    stride >>= 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // TODO - Paul 2014.01.24. I removed this unrolling and changed the above loop
  // from (stride >32) to ensure compatibility with CPU platforms.
  // once I can work out how to use the preferred_work_group_size_multiple that
  // Tom suggested, I'll re-add the optimized version.
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

// Paul - Implementation of double atomicAdd from CUDA Programming Guide:
// Appendix B.12
// ported to OpenCL
#if 0
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

double atomicAdd_db(__global double *address, __global int *loc, double val) {
  double old;

  while (atomic_xchg(loc, 1))
    ;
  old = *address;
  *address += val;
  atomic_xchg(loc, 0);

  return old;
}

unsigned long atomicAdd_ul(__global unsigned long *address, __global int *loc,
                           unsigned long val) {
  unsigned long old;

  while (atomic_xchg(loc, 1))
    ;
  old = *address;
  *address += val;
  atomic_xchg(loc, 0);

  return old;
}

double atomicAdd_fl(__global float *address, float val) {
  __global unsigned int *address_as_ui = (__global unsigned int *)address;
  // unsigned long old = *address_as_ul, assumed;
  volatile unsigned int old = atomic_add(address_as_ui, 0), assumed;
  do {
    assumed = old;
    old = atomic_cmpxchg(address_as_ui, assumed,
                         as_uint(val + as_float(assumed)));
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
                                int i, int j, int k, int sx, int sy, int sz,
                                int ex, int ey, int ez, int gz,
                                __global double *reduction, int len_,
                                __local volatile double *psum,
                                __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));
  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi1[x + y * i + z * i * j] * phi2[x + y * i + z * i * j];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_db(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomicAdd_db(reduction, loc, psum[0]);
}

__kernel void kernel_dotProd_fl(__global float *phi1, __global float *phi2,
                                int i, int j, int k, int sx, int sy, int sz,
                                int ex, int ey, int ez, int gz,
                                __global float *reduction, int len_,
                                __local volatile float *psum,
                                __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));
  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi1[x + y * i + z * i * j] * phi2[x + y * i + z * i * j];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_fl(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomicAdd_fl(reduction, psum[0]);
}

__kernel void
kernel_dotProd_ul(__global unsigned long *phi1, __global unsigned long *phi2,
                  int i, int j, int k, int sx, int sy, int sz, int ex, int ey,
                  int ez, int gz, __global unsigned long *reduction, int len_,
                  __local volatile unsigned long *psum, __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));
  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi1[x + y * i + z * i * j] * phi2[x + y * i + z * i * j];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_ul(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomicAdd_ul(reduction, loc, psum[0]);
}

__kernel void kernel_dotProd_in(__global int *phi1, __global int *phi2, int i,
                                int j, int k, int sx, int sy, int sz, int ex,
                                int ey, int ez, int gz, __global int *reduction,
                                int len_, __local volatile int *psum,
                                __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));
  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi1[x + y * i + z * i * j] * phi2[x + y * i + z * i * j];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_in(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomic_add(reduction, psum[0]);
}

__kernel void
kernel_dotProd_ui(__global unsigned int *phi1, __global unsigned int *phi2,
                  int i, int j, int k, int sx, int sy, int sz, int ex, int ey,
                  int ez, int gz, __global unsigned int *reduction, int len_,
                  __local volatile unsigned int *psum, __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));
  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi1[x + y * i + z * i * j] * phi2[x + y * i + z * i * j];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_ui(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomic_add(reduction, psum[0]);
}

__kernel void kernel_reduce_db(__global double *phi, int i, int j, int k,
                               int sx, int sy, int sz, int ex, int ey, int ez,
                               int gz, __global double *reduction, int len_,
                               __local volatile double *psum,
                               __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));

  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi[x + y * i + z * i * j];
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_db(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomicAdd_db(reduction, loc, psum[0]);
}

__kernel void kernel_reduce_fl(__global float *phi, int i, int j, int k, int sx,
                               int sy, int sz, int ex, int ey, int ez, int gz,
                               __global float *reduction, int len_,
                               __local volatile float *psum,
                               __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));

  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi[x + y * i + z * i * j];
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_fl(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomicAdd_fl(reduction, psum[0]);
}

__kernel void kernel_reduce_ul(__global unsigned long *phi, int i, int j, int k,
                               int sx, int sy, int sz, int ex, int ey, int ez,
                               int gz, __global unsigned long *reduction,
                               int len_, __local volatile unsigned long *psum,
                               __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));

  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi[x + y * i + z * i * j];
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_ul(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomicAdd_ul(reduction, loc, psum[0]);
}

__kernel void kernel_reduce_in(__global int *phi, int i, int j, int k, int sx,
                               int sy, int sz, int ex, int ey, int ez, int gz,
                               __global int *reduction, int len_,
                               __local volatile int *psum, __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));

  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi[x + y * i + z * i * j];
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_in(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomic_add(reduction, psum[0]);
}

__kernel void kernel_reduce_ui(__global unsigned int *phi, int i, int j, int k,
                               int sx, int sy, int sz, int ex, int ey, int ez,
                               int gz, __global unsigned int *reduction,
                               int len_, __local volatile unsigned int *psum,
                               __global int *loc) {

  int tid, loads, x, y, z, itr;
  int boundx, boundy, boundz;
  tid = get_local_id(0) + (get_local_id(1)) * get_local_size(0) +
        (get_local_id(2)) * (get_local_size(0) * get_local_size(1));

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  loads = gz;

  psum[tid] = 0;
  boundy = ((y >= sy) && (y <= ey));
  boundx = ((x >= sx) && (x <= ex));

  for (itr = 0; itr < loads; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z >= sz) && (z <= ez));
    if (boundx && boundy && boundz)
      psum[tid] += phi[x + y * i + z * i * j];
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  block_reduction_ui(psum, tid, len_);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (tid == 0)
    atomic_add(reduction, psum[0]);
}

__kernel void kernel_transpose_2d_db(__global double *odata,
                                     __global double *idata, int arr_width,
                                     int arr_height, int tran_width,
                                     int tran_height, __local double *tile) {
  //    __local double tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

  int blockIdx_x, blockIdx_y;
  int gridDim_x, gridDim_y;

  // do diagonal reordering
  // The if case degenerates to the else case, no need to have both
  // if (width == height)
  //{
  //    blockIdx_y = get_group_id(0);
  //    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
  //}
  // else
  //{
  // First figure out your number among the actual grid blocks
  int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1);
  // Then figure out how many logical blocks are required in each dimension
  gridDim_x = (tran_width - 1 + get_local_size(0)) / get_local_size(0);
  gridDim_y = (tran_height - 1 + get_local_size(1)) / get_local_size(1);
  // Then how many logical and actual grid blocks
  int logicalBlocks = gridDim_x * gridDim_y;
  int gridBlocks = get_num_groups(0) * get_num_groups(1);
  // Loop over all logical blocks
  for (; bid < logicalBlocks; bid += gridBlocks) {
    // Compute the current logical block index in each dimension
    blockIdx_y = bid % gridDim_y;
    blockIdx_x = ((bid / gridDim_y) + blockIdx_y) % gridDim_x;
    //}

    // int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
    // int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
    // int index_in = xIndex_in + (yIndex_in)*width;
    int index_in = xIndex_in + (yIndex_in)*arr_width;

    // int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
    // int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
    // int index_out = xIndex_out + (yIndex_out)*height;
    int index_out = xIndex_out + (yIndex_out)*arr_height;

    if (xIndex_in < tran_width && yIndex_in < tran_height)
      // tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
      tile[get_local_id(1) * (get_local_size(0)) + get_local_id(0)] =
          idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // if(xIndex_out < width && yIndex_out < height)
    if (xIndex_out < tran_height && yIndex_out < tran_width)
      // odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
      odata[index_out] =
          tile[get_local_id(1) + (get_local_size(1)) * get_local_id(0)];

    // Added with the loop to ensure writes are finished before new vals go into
    // shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}

__kernel void kernel_transpose_2d_fl(__global float *odata,
                                     __global float *idata, int arr_width,
                                     int arr_height, int tran_width,
                                     int tran_height, __local float *tile) {
  //    __local float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

  int blockIdx_x, blockIdx_y;
  int gridDim_x, gridDim_y;

  // do diagonal reordering
  // The if case degenerates to the else case, no need to have both
  // if (width == height)
  //{
  //    blockIdx_y = get_group_id(0);
  //    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
  //}
  // else
  //{
  // First figure out your number among the actual grid blocks
  int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1);
  // Then figure out how many logical blocks are required in each dimension
  gridDim_x = (tran_width - 1 + get_local_size(0)) / get_local_size(0);
  gridDim_y = (tran_height - 1 + get_local_size(1)) / get_local_size(1);
  // Then how many logical and actual grid blocks
  int logicalBlocks = gridDim_x * gridDim_y;
  int gridBlocks = get_num_groups(0) * get_num_groups(1);
  // Loop over all logical blocks
  for (; bid < logicalBlocks; bid += gridBlocks) {
    // Compute the current logical block index in each dimension
    blockIdx_y = bid % gridDim_y;
    blockIdx_x = ((bid / gridDim_y) + blockIdx_y) % gridDim_x;
    //}

    // int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
    // int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
    // int index_in = xIndex_in + (yIndex_in)*width;
    int index_in = xIndex_in + (yIndex_in)*arr_width;

    // int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
    // int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
    // int index_out = xIndex_out + (yIndex_out)*height;
    int index_out = xIndex_out + (yIndex_out)*arr_height;

    if (xIndex_in < tran_width && yIndex_in < tran_height)
      // tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
      tile[get_local_id(1) * (get_local_size(0)) + get_local_id(0)] =
          idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // if(xIndex_out < width && yIndex_out < height)
    if (xIndex_out < tran_height && yIndex_out < tran_width)
      // odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
      odata[index_out] =
          tile[get_local_id(1) + (get_local_size(1)) * get_local_id(0)];

    // Added with the loop to ensure writes are finished before new vals go into
    // shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}

__kernel void kernel_transpose_2d_ul(__global unsigned long *odata,
                                     __global unsigned long *idata,
                                     int arr_width, int arr_height,
                                     int tran_width, int tran_height,
                                     __local unsigned long *tile) {
  //    __local unsigned long tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

  int blockIdx_x, blockIdx_y;
  int gridDim_x, gridDim_y;

  // do diagonal reordering
  // The if case degenerates to the else case, no need to have both
  // if (width == height)
  //{
  //    blockIdx_y = get_group_id(0);
  //    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
  //}
  // else
  //{
  // First figure out your number among the actual grid blocks
  int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1);
  // Then figure out how many logical blocks are required in each dimension
  gridDim_x = (tran_width - 1 + get_local_size(0)) / get_local_size(0);
  gridDim_y = (tran_height - 1 + get_local_size(1)) / get_local_size(1);
  // Then how many logical and actual grid blocks
  int logicalBlocks = gridDim_x * gridDim_y;
  int gridBlocks = get_num_groups(0) * get_num_groups(1);
  // Loop over all logical blocks
  for (; bid < logicalBlocks; bid += gridBlocks) {
    // Compute the current logical block index in each dimension
    blockIdx_y = bid % gridDim_y;
    blockIdx_x = ((bid / gridDim_y) + blockIdx_y) % gridDim_x;
    //}

    // int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
    // int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
    // int index_in = xIndex_in + (yIndex_in)*width;
    int index_in = xIndex_in + (yIndex_in)*arr_width;

    // int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
    // int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
    // int index_out = xIndex_out + (yIndex_out)*height;
    int index_out = xIndex_out + (yIndex_out)*arr_height;

    if (xIndex_in < tran_width && yIndex_in < tran_height)
      // tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
      tile[get_local_id(1) * (get_local_size(0)) + get_local_id(0)] =
          idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // if(xIndex_out < width && yIndex_out < height)
    if (xIndex_out < tran_height && yIndex_out < tran_width)
      // odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
      odata[index_out] =
          tile[get_local_id(1) + (get_local_size(1)) * get_local_id(0)];

    // Added with the loop to ensure writes are finished before new vals go into
    // shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}

__kernel void kernel_transpose_2d_in(__global int *odata, __global int *idata,
                                     int arr_width, int arr_height,
                                     int tran_width, int tran_height,
                                     __local int *tile) {
  //    __local int tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

  int blockIdx_x, blockIdx_y;
  int gridDim_x, gridDim_y;

  // do diagonal reordering
  // The if case degenerates to the else case, no need to have both
  // if (width == height)
  //{
  //    blockIdx_y = get_group_id(0);
  //    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
  //}
  // else
  //{
  // First figure out your number among the actual grid blocks
  int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1);
  // Then figure out how many logical blocks are required in each dimension
  gridDim_x = (tran_width - 1 + get_local_size(0)) / get_local_size(0);
  gridDim_y = (tran_height - 1 + get_local_size(1)) / get_local_size(1);
  // Then how many logical and actual grid blocks
  int logicalBlocks = gridDim_x * gridDim_y;
  int gridBlocks = get_num_groups(0) * get_num_groups(1);
  // Loop over all logical blocks
  for (; bid < logicalBlocks; bid += gridBlocks) {
    // Compute the current logical block index in each dimension
    blockIdx_y = bid % gridDim_y;
    blockIdx_x = ((bid / gridDim_y) + blockIdx_y) % gridDim_x;
    //}

    // int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
    // int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
    // int index_in = xIndex_in + (yIndex_in)*width;
    int index_in = xIndex_in + (yIndex_in)*arr_width;

    // int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
    // int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
    // int index_out = xIndex_out + (yIndex_out)*height;
    int index_out = xIndex_out + (yIndex_out)*arr_height;

    if (xIndex_in < tran_width && yIndex_in < tran_height)
      // tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
      tile[get_local_id(1) * (get_local_size(0)) + get_local_id(0)] =
          idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // if(xIndex_out < width && yIndex_out < height)
    if (xIndex_out < tran_height && yIndex_out < tran_width)
      // odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
      odata[index_out] =
          tile[get_local_id(1) + (get_local_size(1)) * get_local_id(0)];

    // Added with the loop to ensure writes are finished before new vals go into
    // shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}

__kernel void kernel_transpose_2d_ui(__global unsigned int *odata,
                                     __global unsigned int *idata,
                                     int arr_width, int arr_height,
                                     int tran_width, int tran_height,
                                     __local unsigned int *tile) {
  //    __local unsigned int tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

  int blockIdx_x, blockIdx_y;
  int gridDim_x, gridDim_y;

  // do diagonal reordering
  // The if case degenerates to the else case, no need to have both
  // if (width == height)
  //{
  //    blockIdx_y = get_group_id(0);
  //    blockIdx_x = (get_group_id(0)+get_group_id(1))%get_num_groups(0);
  //}
  // else
  //{
  // First figure out your number among the actual grid blocks
  int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1);
  // Then figure out how many logical blocks are required in each dimension
  gridDim_x = (tran_width - 1 + get_local_size(0)) / get_local_size(0);
  gridDim_y = (tran_height - 1 + get_local_size(1)) / get_local_size(1);
  // Then how many logical and actual grid blocks
  int logicalBlocks = gridDim_x * gridDim_y;
  int gridBlocks = get_num_groups(0) * get_num_groups(1);
  // Loop over all logical blocks
  for (; bid < logicalBlocks; bid += gridBlocks) {
    // Compute the current logical block index in each dimension
    blockIdx_y = bid % gridDim_y;
    blockIdx_x = ((bid / gridDim_y) + blockIdx_y) % gridDim_x;
    //}

    // int xIndex_in = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_in = blockIdx_x * get_local_size(0) + get_local_id(0);
    // int yIndex_in = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_in = blockIdx_y * get_local_size(1) + get_local_id(1);
    // int index_in = xIndex_in + (yIndex_in)*width;
    int index_in = xIndex_in + (yIndex_in)*arr_width;

    // int xIndex_out = blockIdx_y * TRANSPOSE_TILE_DIM + get_local_id(0);
    int xIndex_out = blockIdx_y * get_local_size(1) + get_local_id(0);
    // int yIndex_out = blockIdx_x * TRANSPOSE_TILE_DIM + get_local_id(1);
    int yIndex_out = blockIdx_x * get_local_size(0) + get_local_id(1);
    // int index_out = xIndex_out + (yIndex_out)*height;
    int index_out = xIndex_out + (yIndex_out)*arr_height;

    if (xIndex_in < tran_width && yIndex_in < tran_height)
      // tile[get_local_id(1)][get_local_id(0)] =  idata[index_in];
      tile[get_local_id(1) * (get_local_size(0)) + get_local_id(0)] =
          idata[index_in];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // if(xIndex_out < width && yIndex_out < height)
    if (xIndex_out < tran_height && yIndex_out < tran_width)
      // odata[index_out] = tile[get_local_id(0)][get_local_id(1)];
      odata[index_out] =
          tile[get_local_id(1) + (get_local_size(1)) * get_local_id(0)];

    // Added with the loop to ensure writes are finished before new vals go into
    // shared memory
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}
int get_pack_index(int tid, __local int *a, int start, int count,
                   __constant int *c_face_size, __constant int *c_face_stride,
                   __constant int *c_face_child_size) {
  int i, j, k, l;
  int pos;
  for (i = 0; i < count; i++)
    a[tid % get_local_size(0) + i * get_local_size(0)] = 0;

  for (i = 0; i < count; i++) {
    k = 0;
    for (j = 0; j < i; j++) {
      k += a[tid % get_local_size(0) + j * get_local_size(0)] *
           c_face_child_size[j];
    }
    l = c_face_child_size[i];
    for (j = 0; j < c_face_size[i]; j++) {
      if (tid - k < l)
        break;
      else
        l += c_face_child_size[i];
    }
    a[tid % get_local_size(0) + i * get_local_size(0)] = j;
  }
  pos = start;
  for (i = 0; i < count; i++) {
    pos +=
        a[tid % get_local_size(0) + i * get_local_size(0)] * c_face_stride[i];
  }
  return pos;
}

__kernel void kernel_pack_db(__global double *packed_buf, __global double *buf,
                             int size, int start, int count,
                             __constant int *c_face_size,
                             __constant int *c_face_stride,
                             __constant int *c_face_child_size,
                             __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size,
                                         c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_fl(__global float *packed_buf, __global float *buf,
                             int size, int start, int count,
                             __constant int *c_face_size,
                             __constant int *c_face_stride,
                             __constant int *c_face_child_size,
                             __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size,
                                         c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_ul(__global unsigned long *packed_buf,
                             __global unsigned long *buf, int size, int start,
                             int count, __constant int *c_face_size,
                             __constant int *c_face_stride,
                             __constant int *c_face_child_size,
                             __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size,
                                         c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_in(__global int *packed_buf, __global int *buf,
                             int size, int start, int count,
                             __constant int *c_face_size,
                             __constant int *c_face_stride,
                             __constant int *c_face_child_size,
                             __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size,
                                         c_face_stride, c_face_child_size)];
}

__kernel void kernel_pack_ui(__global unsigned int *packed_buf,
                             __global unsigned int *buf, int size, int start,
                             int count, __constant int *c_face_size,
                             __constant int *c_face_stride,
                             __constant int *c_face_child_size,
                             __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    packed_buf[idx] = buf[get_pack_index(idx, a, start, count, c_face_size,
                                         c_face_stride, c_face_child_size)];
}

__kernel void kernel_unpack_db(__global double *packed_buf,
                               __global double *buf, int size, int start,
                               int count, __constant int *c_face_size,
                               __constant int *c_face_stride,
                               __constant int *c_face_child_size,
                               __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride,
                       c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_fl(__global float *packed_buf, __global float *buf,
                               int size, int start, int count,
                               __constant int *c_face_size,
                               __constant int *c_face_stride,
                               __constant int *c_face_child_size,
                               __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride,
                       c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_ul(__global unsigned long *packed_buf,
                               __global unsigned long *buf, int size, int start,
                               int count, __constant int *c_face_size,
                               __constant int *c_face_stride,
                               __constant int *c_face_child_size,
                               __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride,
                       c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_in(__global int *packed_buf, __global int *buf,
                               int size, int start, int count,
                               __constant int *c_face_size,
                               __constant int *c_face_stride,
                               __constant int *c_face_child_size,
                               __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride,
                       c_face_child_size)] = packed_buf[idx];
}

__kernel void kernel_unpack_ui(__global unsigned int *packed_buf,
                               __global unsigned int *buf, int size, int start,
                               int count, __constant int *c_face_size,
                               __constant int *c_face_stride,
                               __constant int *c_face_child_size,
                               __local int *a) {
  int idx = get_global_id(0);
  const int nthreads = get_global_size(0);
  // this loop handles both nthreads > size and nthreads < size
  for (; idx < size; idx += nthreads)
    buf[get_pack_index(idx, a, start, count, c_face_size, c_face_stride,
                       c_face_child_size)] = packed_buf[idx];
}
// this kernel works for 3D data only.
//  i,j,k are the array dimensions
//  s* parameters are start values in each dimension.
//  e* parameters are end values in each dimension.
//  s* and e* are only necessary when the halo layers
//    has different thickness along various directions.
//  len_ is number of threads in a threadblock.
//       This can be computed in the kernel itself.

// work with 2D and 3D thread blocks
__kernel void kernel_stencil_3d7p_db(const __global double *__restrict__ ind,
                                     __global double *__restrict__ outd, int i,
                                     int j, int k, int sx, int sy, int sz,
                                     int ex, int ey, int ez, int gz, int len_,
                                     __local double *bind) {
  int x, y, z, itr;
  bool boundx, boundy, boundz;

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  boundy = ((y > sy) && (y < ey));
  boundx = ((x > sx) && (x < ex));

  for (itr = 0; itr < gz; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z > sz) && (z < ez));
    if (boundx && boundy && boundz)
      outd[x + y * i + z * i * j] =
          (ind[x + y * i + (z - 1) * i * j] + ind[(x - 1) + y * i + z * i * j] +
           ind[x + (y - 1) * i + z * i * j] + ind[x + y * i + z * i * j] +
           ind[x + (y + 1) * i + z * i * j] + ind[(x + 1) + y * i + z * i * j] +
           ind[x + y * i + (z + 1) * i * j]) /
          (double)7;
  }
}

#if 0
//Read-only cache + Rigster blocking (Z) + smem blocking (X-Y)
// work only with 2D thread blocks (use rectangular blocks, i.e. 64*4, 128*2)
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

//Read-only cache + Rigster blocking (Z)
// work only with 2D thread blocks
__kernel void kernel_stencil_3d7p_db_v1(const __global double * __restrict__ ind, __global double * __restrict__ outd,
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

// work with 2D and 3D thread blocks
__kernel void kernel_stencil_3d7p_fl(const __global float *__restrict__ ind,
                                     __global float *__restrict__ outd, int i,
                                     int j, int k, int sx, int sy, int sz,
                                     int ex, int ey, int ez, int gz, int len_,
                                     __local float *bind) {
  int x, y, z, itr;
  bool boundx, boundy, boundz;

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  boundy = ((y > sy) && (y < ey));
  boundx = ((x > sx) && (x < ex));

  for (itr = 0; itr < gz; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z > sz) && (z < ez));
    if (boundx && boundy && boundz)
      outd[x + y * i + z * i * j] =
          (ind[x + y * i + (z - 1) * i * j] + ind[(x - 1) + y * i + z * i * j] +
           ind[x + (y - 1) * i + z * i * j] + ind[x + y * i + z * i * j] +
           ind[x + (y + 1) * i + z * i * j] + ind[(x + 1) + y * i + z * i * j] +
           ind[x + y * i + (z + 1) * i * j]) /
          (float)7;
  }
}

// work with 2D and 3D thread blocks
__kernel void
kernel_stencil_3d7p_ul(const __global unsigned long *__restrict__ ind,
                       __global unsigned long *__restrict__ outd, int i, int j,
                       int k, int sx, int sy, int sz, int ex, int ey, int ez,
                       int gz, int len_, __local unsigned long *bind) {
  int x, y, z, itr;
  bool boundx, boundy, boundz;

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  boundy = ((y > sy) && (y < ey));
  boundx = ((x > sx) && (x < ex));

  for (itr = 0; itr < gz; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z > sz) && (z < ez));
    if (boundx && boundy && boundz)
      outd[x + y * i + z * i * j] =
          (ind[x + y * i + (z - 1) * i * j] + ind[(x - 1) + y * i + z * i * j] +
           ind[x + (y - 1) * i + z * i * j] + ind[x + y * i + z * i * j] +
           ind[x + (y + 1) * i + z * i * j] + ind[(x + 1) + y * i + z * i * j] +
           ind[x + y * i + (z + 1) * i * j]) /
          (unsigned long)7;
  }
}

// work with 2D and 3D thread blocks
__kernel void kernel_stencil_3d7p_in(const __global int *__restrict__ ind,
                                     __global int *__restrict__ outd, int i,
                                     int j, int k, int sx, int sy, int sz,
                                     int ex, int ey, int ez, int gz, int len_,
                                     __local int *bind) {
  int x, y, z, itr;
  bool boundx, boundy, boundz;

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  boundy = ((y > sy) && (y < ey));
  boundx = ((x > sx) && (x < ex));

  for (itr = 0; itr < gz; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z > sz) && (z < ez));
    if (boundx && boundy && boundz)
      outd[x + y * i + z * i * j] =
          (ind[x + y * i + (z - 1) * i * j] + ind[(x - 1) + y * i + z * i * j] +
           ind[x + (y - 1) * i + z * i * j] + ind[x + y * i + z * i * j] +
           ind[x + (y + 1) * i + z * i * j] + ind[(x + 1) + y * i + z * i * j] +
           ind[x + y * i + (z + 1) * i * j]) /
          (int)7;
  }
}

// work with 2D and 3D thread blocks
__kernel void
kernel_stencil_3d7p_ui(const __global unsigned int *__restrict__ ind,
                       __global unsigned int *__restrict__ outd, int i, int j,
                       int k, int sx, int sy, int sz, int ex, int ey, int ez,
                       int gz, int len_, __local int *bind) {
  int x, y, z, itr;
  bool boundx, boundy, boundz;

  x = (get_group_id(0)) * get_local_size(0) + get_local_id(0) + sx;
  y = (get_group_id(1)) * get_local_size(1) + get_local_id(1) + sy;

  boundy = ((y > sy) && (y < ey));
  boundx = ((x > sx) && (x < ex));

  for (itr = 0; itr < gz; itr++) {
    z = itr * get_local_size(2) + get_local_id(2) + sz;
    boundz = ((z > sz) && (z < ez));
    if (boundx && boundy && boundz)
      outd[x + y * i + z * i * j] =
          (ind[x + y * i + (z - 1) * i * j] + ind[(x - 1) + y * i + z * i * j] +
           ind[x + (y - 1) * i + z * i * j] + ind[x + y * i + z * i * j] +
           ind[x + (y + 1) * i + z * i * j] + ind[(x + 1) + y * i + z * i * j] +
           ind[x + y * i + (z + 1) * i * j]) /
          (unsigned int)7;
  }
}
