#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
   void block_reduction(__local double *psum, int tid, int len_) {
      //implicit none
      //real(8), shared,dimension(len_)::psum
      //integer::tid,stride,len_
	int stride = len_ >> 1; //STRIDE = len_/2
            while (stride > 32) {
             //DO WHILE(STRIDE.GT.32)
		if (tid +stride < len_) psum[tid] += psum[tid+stride];
                  //IF(TID.LE.STRIDE) THEN
                  //      PSUM(TID) = PSUM(TID)+PSUM(TID+STRIDE)
                  //END IF
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //  CALL SYNCTHREADS()
                stride >>= 1;//  STRIDE = STRIDE/2
            } //END DO
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()      
		//! this is for unrolling.      
            if (tid <= 32) { //IF(TID.LE.32) THEN
                   psum[tid] += psum[tid+32]; //PSUM(TID) = PSUM(TID) + PSUM(TID+32)
                   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+16]; //PSUM(TID) = PSUM(TID) + PSUM(TID+16)
                   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+8]; //PSUM(TID) = PSUM(TID) + PSUM(TID+8)
                   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+4]; //PSUM(TID) = PSUM(TID) + PSUM(TID+4)
                   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+2]; //PSUM(TID) = PSUM(TID) + PSUM(TID+2)
                   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+1]; //PSUM(TID) = PSUM(TID) + PSUM(TID+1)
                   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()
           } //END IF
  } //end subroutine block_reduction

//Paul - Implementation of double atomicAdd from CUDA Programming Guide: Appendix B.12
 double atomicAdd(__global double* address, double val)
{
    __global unsigned long * address_as_ul =
                              (__global unsigned long *)address;
    unsigned long old = *address_as_ul, assumed;
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
//  ATTRIBUTES(GLOBAL) &
//     & SUBROUTINE KERNEL_REDUCTION3(PHI1,PHI2,j,i,k,sx,sy,sz,ex,ey,ez,gz,&
//     & REDUCTION,len_)
	__kernel void kernel_reduction3(__global double *phi1, __global double *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez, 
		int gz, __global double * reduction, int len_, __local double * psum) {
            //IMPLICIT NONE
            //real(8), DEVICE, DIMENSION(i,j,k):: PHI1,PHI2
//!            double[32] par_sum; //real(8), DEVICE, DIMENSION(32):: PAR_SUM
            //extern __local double psum[]; //real(8),VOLATILE, SHARED, DIMENSION(len_)::PSUM
            //real(8), DEVICE:: REDUCTION
            int stride, istat, tid, loads, x, y, z, itr; //INTEGER:: STRIDE,ISTAT,TID,loads,x,y,z,itr
            //INTEGER, VALUE:: i,j,k,sx,sy,sz,len_,ex,ey,ez,gz
            int boundx, boundy, boundz; //logical::  boundx, boundy, boundz
            tid = get_local_id(0)+(get_local_id(1))*get_local_size(0)+(get_local_id(2))*(get_local_size(0)*get_local_size(1)); //TID = THREADIDX%X+(threadidx%y-1)*blockdim%x &
//     & +(threadidx%z-1)*blockdim%x*blockdim%y

            x = (get_group_id(0))*get_local_size(0)+get_local_id(0)+sx ; //x = (blockidx%x-1)*blockdim%x + threadidx%x + sx -1
            y = (get_group_id(1))*get_local_size(1)+get_local_id(1)+sy; //y = (blockidx%y-1)*blockdim%y + threadidx%y + sy -1
            
            loads = gz;
            
            psum[tid] = 0;
            boundy = ((y >= sy) && (y <= ey)); //y.ge.(sy).and.y.le.(ey)
            boundx = ((x >= sx) && (x <= ex)); //x.ge.(sx).and.x.le.(ex)
            
            for (itr = 0; itr < loads; itr++) { //do itr=1,loads
                  z = itr*get_local_size(0)+get_local_id(2) +sz; //z = (itr-1)*blockdim%z+threadidx%z + sz -1
                  boundz = ((z >= sz) && (z <= ez)); //z.ge.(sz).and.z.le.(ez)
                  if (boundx && boundy && boundz) psum[tid] += 1;//phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j]; ////{ if(boundx.and.boundy.and.boundz) then
                  //      PSUM(TID) = psum(tid) + PHI1(y,x,z) * PHI2(Y,X,Z)
                  //end if
            } //end do
            
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()
            block_reduction(psum,tid,len_); //call block_reduction(psum,tid,stride,len_)
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //CALL SYNCTHREADS()            


            if(tid == 0) istat = atomicAdd(reduction,psum[0]); //IF(TID.EQ.1) THEN
            //      ISTAT = ATOMICADD(REDUCTION,PSUM(1))
            //END IF
       }



