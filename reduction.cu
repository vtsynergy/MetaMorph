#include <stdio.h>
//  module kernels
//      use cudafor
//  contains
//  ! this reduces elements in a thread block.
//  attributes(device) subroutine block_reduction(psum,tid,stride,len_)
  __device__ void block_reduction(double *psum, int tid, int len_) {
      //implicit none
      //real(8), shared,dimension(len_)::psum
      //integer::tid,stride,len_
	int stride = len_ >> 1; //STRIDE = len_/2
            while (stride > 32) {
             //DO WHILE(STRIDE.GT.32)
		if (tid < stride) psum[tid] += psum[tid+stride];
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
            int stride, istat, tid, loads, x, y, z, itr; //INTEGER:: STRIDE,ISTAT,TID,loads,x,y,z,itr
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
                  z = itr*blockDim.x+threadIdx.z +sz; //z = (itr-1)*blockdim%z+threadidx%z + sz -1
                  boundz = ((z >= sz) && (z <= ez)); //z.ge.(sz).and.z.le.(ez)
                  if (boundx && boundy && boundz) psum[tid] += phi1[x+y*i+z*i*j] * phi2[x+y*i+z*i*j]; ////{ if(boundx.and.boundy.and.boundz) then
                  //      PSUM(TID) = psum(tid) + PHI1(y,x,z) * PHI2(Y,X,Z)
                  //end if
            } //end do
            
            __syncthreads(); //CALL SYNCTHREADS()
            block_reduction(psum,tid,len_); //call block_reduction(psum,tid,stride,len_)
            __syncthreads(); //CALL SYNCTHREADS()            


            if(tid == 0) istat = atomicAdd(reduction,psum[0]); //IF(TID.EQ.1) THEN
            //      ISTAT = ATOMICADD(REDUCTION,PSUM(1))
            //END IF
       } //END SUBROUTINE KERNEL_REDUCTION3

//! this assumes the data to be 4D but does only for on sub 3D array.
//      end module kernels
//      module dims
            int ni, nj, nk, nm; //integer::ni,nj,nk,nm
//      end module dims
//      module data_
//            use cudafor
            
//            implicit none
            
            double *dev_data3, *dev_data3_2; //real(8), allocatable,device,dimension(:,:,:)::dev_data3,dev_data3_2
            double *dev_data4; //real(8), allocatable,device,dimension(:,:,:,:)::dev_data4
            double *data3; //real(8), allocatable, dimension(:,:,:)::data3
            double *data4; //real(8), allocatable, dimension(:,:,:,:)::data4
            double *reduction; //real(8), device::reduction
            //integer::ni,nj,nk,nm

//!            allocate(dev_data3(1,1,1))
//!            print *,"allocated"
      
//      end module data_

//      !This does the host and device data allocations.
//      subroutine data_allocate(i,j,k,m)
//            use data_
	void data_allocate(int i, int j, int k, int m) {
//            implicit none
//            integer,intent(in)::i,j,k,m
            int istat = 0; //integer::istat
//!            ni = i;
//!            nj = j;
//!            nk = k;
//!            nm = m;
            printf("ni:\t%d\n", ni); //print *,"ni:",ni
            printf("nj:\t%d\n", nj); //print *,"nj:",nj
            printf("nk:\t%d\n", nk); //print *,"nk:",nk
            data3 = (double *) malloc(sizeof(double)*ni*nj*nk);
	    data4 = (double *) malloc (sizeof(double)*ni*nj*nk*nm); //allocate(data3(ni,nj,nk),data4(ni,nj,nk,nm),stat=istat)
            printf("Status:\t%d\n", istat); //NOOP for output compatibility //print *,"Status:",istat
            istat |= cudaMalloc((void **) &dev_data3, sizeof(double)*ni*nj*nk);
	    istat |= cudaMalloc((void **) &dev_data3_2, sizeof(double)*ni*nj*nk);
	    istat |= cudaMalloc((void **) &dev_data4, sizeof(double)*ni*nj*nk*nm); //allocate(dev_data3(ni,nj,nk),dev_data3_2(ni,nj,nk),dev_data4(ni,nj,nk,nm),stat=istat)
	    istat |= cudaMalloc((void **) &reduction, sizeof(double));
            printf("Status:\t%d\n", istat); //print *,"Status:",istat
            printf("Data Allocated\n"); //print *,"Data Allocated"
      } //end subroutine data_allocate

//      !initilialize the host side data that has to be reduced here.
//      !For now I initialized it to 1.0
      void data_initialize() { //subroutine data_initialize
            //use data_
            //implicit none
int i;
	for (i = ni*nj*nk*nm-1; i >= ni*nj*nk; i--) {
            data4[i] = 1.0;
	}
	for (; i >= 0; i--) {
	    data4[i] = 1.0;
            data3[i] = 1.0;
	}
      } //end subroutine data_initialize

//      !Transfers data from host to device
      void data_transfer_h2d() { //subroutine data_transfer_h2d
//            use data_
//            implicit none
            cudaMemcpy((void *) dev_data3, (void *) data3, sizeof(double)*ni*nj*nk, cudaMemcpyHostToDevice); //dev_data3(:,:,:) = data3(:,:,:)
            cudaMemcpy((void *) dev_data4, (void *) data4, sizeof(double)*ni*nj*nk*nm, cudaMemcpyHostToDevice); //dev_data4(:,:,:,:) = data4(:,:,:,:)
            cudaMemcpy((void *) dev_data3_2, (void *) data3, sizeof(double)*ni*nj*nk, cudaMemcpyHostToDevice); //dev_data3_2(:,:,:) = data3(:,:,:)
      } //end subroutine data_transfer_h2d
      void deallocate_() { //subroutine deallocate_
//            use data_
//            implicit none
            cudaFree(dev_data3); //deallocate(dev_data3)
            free(data3); //deallocate(data3)
            cudaFree(dev_data3_2); //deallocate(dev_data3_2)
            cudaFree(dev_data4); //deallocate(dev_data4)
            free(data4); //deallocate(data4)
	    cudaFree(reduction);
      } //end subroutine deallocate_
      void gpu_initialize() { //subroutine gpu_initialize

//            use cudafor

//            implicit none

            int istat, deviceused, idevice = 0; //integer::istat, deviceused, idevice

//            ! Initialize GPU
            istat = cudaSetDevice(idevice);

//            ! cudaChooseDevice
//            ! Tell me which GPU I use
            istat = cudaGetDevice(&deviceused);
            printf("Device used\t%d\n", deviceused); //print *, 'Device used', deviceused
		

      } //end subroutine gpu_initialize
      
      int main(int argc, char **argv) { //program main
//            use kernels
//            use data_
//            use cudafor
//            implicit none
            int tx, ty, tz, gx, gy, gz, istat, i; //integer::tx,ty,tz,gx,gy,gz,istat,i
            dim3 dimgrid, dimblock; //type(dim3)::dimgrid,dimblock
            char args[32]; //character(len=32)::args
            double sum_dot_gpu; //real(8)::sum_dot_gpu
            double *dev_, *dev2; //real(8), allocatable, device, dimension(:,:) :: dev_,DEV2
            i = argc; //i = command_argument_count() 
            if (i < 8) { //if(i.lt.7) then
                  printf("<ni><nj><nk><nm><tblockx><tblocky><tblockz>"); //print *,"<ni><nj><nk><nm><tblockx><tblocky><tblockz>"
                  return(1); //stop
            } //end if
            //call getarg(1,args)
            ni = atoi(argv[1]); //read(args,'(I10)') ni
            //call getarg(2,args)
            nj = atoi(argv[2]); //read(args,'(I10)') nj
            //call getarg(3,args)
            nk = atoi(argv[3]); //read(args,'(I10)') nk
            
            //call getarg(4,args)
            nm = atoi(argv[4]); //read(args,'(I10)') nm
            
            //call getarg(5,args)
            tx = atoi(argv[5]); //read(args,'(I10)') tx
            //call getarg(6,args)
            ty = atoi(argv[6]); //read(args,'(I10)') ty
            //call getarg(7,args)
            tz = atoi(argv[7]); //read(args,'(I10)') tz
            gpu_initialize(); //call gpu_initialize
            data_allocate(ni,nj,nk,nm); //call data_allocate(ni,nj,nk,nm)
            data_initialize(); //call data_initialize
            data_transfer_h2d(); //call data_transfer_h2d
            printf("Performing reduction\n"); //print *,'Performing reduction'
            //dev_data3 = 1.0
//!            tx = 8
//!            ty = 8
//!            tz = 2
//!            gx = 0 
//!            gy = 0
//!            gz = 0
            dimblock = dim3(tx,ty,tz);
            printf("ni:\t%d\n", ni); //print *,"ni:",ni
            printf("nj:\t%d\n", nj); //print *,"nj:",nj
            printf("nk:\t%d\n", nk); //print *,"nk:",nk
            printf("gyr:\t%d\n", (ni-2)%ty); //print *,"gyr:",modulo(ni-2,ty)

            printf("gxr:\t%d\n", (nj-2)%tx); //print *,"gxr:",modulo(ni-2,tx)
            printf("gzr:\t%d\n", (nk-2)%tz); //print *,"gzr:",modulo(nk-2,tz)
            if ((ni-2)%ty != 0)  //if(modulo(ni-2,ty).ne.0)then
                  gy = (ni-2)/ty +1;
            else
                  gy = (ni-2)/ty;
            //end if
            if ((nj-2)%tx != 0) //if(modulo(nj-2,tx).ne.0)then
                  gx = (nj-2)/tx +1;
            else
                  gx = (nj-2)/tx;
            //end if
            if ((nk-2)%tz != 0) //if(modulo(nk-2,tz).ne.0)then
                  gz = (nk-2)/tz +1;
            else
                  gz = (nk-2)/tz;
            //end if
            dimgrid = dim3(gx,gy,1);
            printf("gx:\t%d\n", gx); //print *,"gx:",gx
            printf("gy:\t%d\n", gy); //print *,"gy:",gy
            printf("gz:\t%d\n", gz); //print *,"gz:",gz
            for (i = 0; i < 10; i++) { //do i=1,10
            cudaMemset(reduction, 0.0, sizeof(double)); //*reduction = 0.0;
           kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*sizeof(double)>>>(dev_data3, //call kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*8>>>(dev_data3 &
           dev_data3_2, ni, nj, nk, 1, 1, 1, nj-2, ni-2, nk-2, gz, reduction, tx*ty*tz); //& ,dev_data3_2,ni,nj,nk,2,2,2,nj-1,ni-1,nk-1,gz,reduction,tx*ty*tz)
            istat = cudaThreadSynchronize(); //cudathreadsynchronize()
	//	printf("cudaThreadSynchronize error code:%d\n", istat);            
istat = cudaMemcpy((void *) &sum_dot_gpu, (void *) reduction, sizeof(double), cudaMemcpyDeviceToHost); //sum_dot_gpu = *reduction;
         //   	printf("cudaMemcpy error code:%d\n", istat);
            printf("Test Reduction:\t%f\n", sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
            } //end do
            deallocate_(); //call deallocate_
      } //end program main
