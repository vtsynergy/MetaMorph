
#include <stdio.h>

#include "afosr_cfd_cuda_core.cuh"



  __device__ void block_reduction(double *psum, int tid, int len_) {
      //implicit none
      //real(8), shared,dimension(len_)::psum
      //integer::tid,stride,len_
	int stride = len_ >> 1; //STRIDE = len_/2
            while (stride > 32) {
             //DO WHILE(STRIDE.GT.32)
		if (tid  < stride) psum[tid] += psum[tid+stride];
                  //IF(TID.LE.STRIDE) THEN
                  //      PSUM(TID) = PSUM(TID)+PSUM(TID+STRIDE)
                  //END IF
		__syncthreads(); //  CALL SYNCTHREADS()
                stride >>= 1;//  STRIDE = STRIDE/2
            } //END DO
            __syncthreads(); //CALL SYNCTHREADS()      
		//! this is for unrolling.      
            if (tid < 32) { //IF(TID.LE.32) THEN
                   psum[tid] += psum[tid+32]; //PSUM(TID) = PSUM(TID) + PSUM(TID+32)
                   __syncthreads(); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+16]; //PSUM(TID) = PSUM(TID) + PSUM(TID+16)
                   __syncthreads(); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+8]; //PSUM(TID) = PSUM(TID) + PSUM(TID+8)
                   __syncthreads(); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+4]; //PSUM(TID) = PSUM(TID) + PSUM(TID+4)
                   __syncthreads(); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+2]; //PSUM(TID) = PSUM(TID) + PSUM(TID+2)
                   __syncthreads(); //CALL SYNCTHREADS()
                   psum[tid] += psum[tid+1]; //PSUM(TID) = PSUM(TID) + PSUM(TID+1)
                   __syncthreads(); //CALL SYNCTHREADS()
           } //END IF
  } //end subroutine block_reduction

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
	__global__ void kernel_reduction3(double *phi1, double *phi2,
		int i, int j, int k,
		int sx, int sy, int sz,
		int ex, int ey, int ez, 
		int gz, double * reduction, int len_) {
            //IMPLICIT NONE
            //real(8), DEVICE, DIMENSION(i,j,k):: PHI1,PHI2
//!            double[32] par_sum; //real(8), DEVICE, DIMENSION(32):: PAR_SUM
            extern __shared__ double psum[]; //real(8),VOLATILE, SHARED, DIMENSION(len_)::PSUM
            //real(8), DEVICE:: REDUCTION
            int tid, loads, x, y, z, itr; //INTEGER:: STRIDE,ISTAT,TID,loads,x,y,z,itr
            //INTEGER, VALUE:: i,j,k,sx,sy,sz,len_,ex,ey,ez,gz
            bool boundx, boundy, boundz; //logical::  boundx, boundy, boundz
            tid = threadIdx.x+(threadIdx.y)*blockDim.x+(threadIdx.z)*(blockDim.x*blockDim.y); //TID = THREADIDX%X+(threadidx%y-1)*blockdim%x &
//     & +(threadidx%z-1)*blockdim%x*blockdim%y

            x = (blockIdx.x)*blockDim.x+threadIdx.x+sx; //x = (blockidx%x-1)*blockdim%x + threadidx%x + sx -1
            y = (blockIdx.y)*blockDim.y+threadIdx.y+sy; //y = (blockidx%y-1)*blockdim%y + threadidx%y + sy -1
            
            loads = gz;
            
            psum[tid] = 0;
            boundy = ((y >= sy) && (y <= ey)); //y.ge.(sy).and.y.le.(ey)
            boundx = ((x >= sx) && (x <= ex)); //x.ge.(sx).and.x.le.(ex)
            
            for (itr = 0; itr < loads; itr++) { //do itr=1,loads
                  z = itr*blockDim.z+threadIdx.z +sz; //z = (itr-1)*blockdim%z+threadidx%z + sz -1
                  boundz = ((z >= sz) && (z <= ez)); //z.ge.(sz).and.z.le.(ez)
                  if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j]; ////{ if(boundx.and.boundy.and.boundz) then
                  //      PSUM(TID) = psum(tid) + PHI1(y,x,z) * PHI2(Y,X,Z)
                  //end if
            } //end do
            
            __syncthreads(); //CALL SYNCTHREADS()
            block_reduction(psum,tid,len_); //call block_reduction(psum,tid,stride,len_)
            __syncthreads(); //CALL SYNCTHREADS()            


            if(tid == 0) atomicAdd(reduction,psum[0]); //IF(TID.EQ.1) THEN
            //      ISTAT = ATOMICADD(REDUCTION,PSUM(1))
            //END IF
       }


cudaError_t cuda_dotProd_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], double * data1, double * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val) {
	cudaError_t ret;
	size_t smem_size = sizeof(double) * (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
	dim3 grid = dim3((*grid_size)[0], (*grid_size)[1], 1);
	dim3 block = dim3((*block_size)[0], (*block_size)[1], (*block_size)[2]);
	//printf("Grid: %d %d %d\n", grid.x, grid.y, grid.z);
	//printf("Block: %d %d %d\n", block.x, block.y, block.z);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_size);
	kernel_reduction3<<<grid,block,smem_size>>>(data1, data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[1], (*arr_end)[0], (*arr_end)[2], (*grid_size)[2], reduced_val, (*block_size)[0] * (*block_size)[1] * (*block_size)[2]);
	//kernel_reduction3<<<grid,block,smem_size>>>(data1, data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[1], (*arr_end)[0], (*arr_end)[2], 8, reduced_val, 8*8*8);
	ret = cudaThreadSynchronize();

	return(ret);
}
