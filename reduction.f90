  module kernels
      use cudafor
  contains
  ! this reduces elements in a thread block.
  attributes(device) subroutine block_reduction(psum,tid,stride,len_)
      implicit none
      real(8), shared,dimension(len_)::psum
      integer::tid,stride,len_
            STRIDE = len_/2
            
             DO WHILE(STRIDE.GT.32)
                  IF(TID.LE.STRIDE) THEN
                        PSUM(TID) = PSUM(TID)+PSUM(TID+STRIDE)
                  END IF
                  CALL SYNCTHREADS()
                  STRIDE = STRIDE/2
            END DO
            CALL SYNCTHREADS()      
		! this is for unrolling.      
            IF(TID.LE.32) THEN
                   PSUM(TID) = PSUM(TID) + PSUM(TID+32)
                   CALL SYNCTHREADS()
                   PSUM(TID) = PSUM(TID) + PSUM(TID+16)
                   CALL SYNCTHREADS()
                   PSUM(TID) = PSUM(TID) + PSUM(TID+8)
                   CALL SYNCTHREADS()
                   PSUM(TID) = PSUM(TID) + PSUM(TID+4)
                   CALL SYNCTHREADS()
                   PSUM(TID) = PSUM(TID) + PSUM(TID+2)
                   CALL SYNCTHREADS()
                   PSUM(TID) = PSUM(TID) + PSUM(TID+1)
                   CALL SYNCTHREADS()
           END IF
  end subroutine block_reduction

  !this kernel works for 3D data only.
  ! PHI1 and PHI2 are input arrays.
  ! s* parameters are start values in each dimension.
  ! e* parameters are end values in each dimension.
  ! s* and e* are only necessary when the halo layers 
  !   has different thickness along various directions.
  ! i,j,k are the array dimensions
  ! len_ is number of threads in a threadblock.
  !      This can be computed in the kernel itself.
  ATTRIBUTES(GLOBAL) &
     & SUBROUTINE KERNEL_REDUCTION3(PHI1,PHI2,j,i,k,sx,sy,sz,ex,ey,ez,gz,&
     & REDUCTION,len_)
            IMPLICIT NONE
            real(8), DEVICE, DIMENSION(i,j,k):: PHI1,PHI2
!            real(8), DEVICE, DIMENSION(32):: PAR_SUM
            real(8),VOLATILE, SHARED, DIMENSION(len_)::PSUM
            real(8), DEVICE:: REDUCTION
            INTEGER:: STRIDE,ISTAT,TID,loads,x,y,z,itr
            INTEGER, VALUE:: i,j,k,sx,sy,sz,len_,ex,ey,ez,gz
            logical::  boundx, boundy, boundz
            TID = THREADIDX%X+(threadidx%y-1)*blockdim%x &
     & +(threadidx%z-1)*blockdim%x*blockdim%y

            x = (blockidx%x-1)*blockdim%x + threadidx%x + sx -1
            y = (blockidx%y-1)*blockdim%y + threadidx%y + sy -1
            
            loads = gz
            
            psum(tid) = 0
            boundy = y.ge.(sy).and.y.le.(ey)
            boundx = x.ge.(sx).and.x.le.(ex)
            
            do itr=1,loads
                  z = (itr-1)*blockdim%z+threadidx%z + sz -1
                  boundz = z.ge.(sz).and.z.le.(ez)
                  if(boundx.and.boundy.and.boundz) then
                        PSUM(TID) = psum(tid) + PHI1(y,x,z) * PHI2(Y,X,Z)
                  end if
            end do
            
            CALL SYNCTHREADS()
            call block_reduction(psum,tid,stride,len_)
            CALL SYNCTHREADS()            

            IF(TID.EQ.1) THEN
                  ISTAT = ATOMICADD(REDUCTION,PSUM(1))
            END IF
       END SUBROUTINE KERNEL_REDUCTION3

     attributes(global) &
    & subroutine kernel_par_sum1_sm(phi,ni,nj,nk,mba,&
    &                              ib,ie,jb,je,kb,ke,m)

       implicit none
       real,device,dimension(ni,nj,nk,mba)::phi
       integer:: ii,jj,kk,i,j,k,index
!      integer :: index_res
       integer,value :: ni,nj,nk,mba,m,ib,jb,kb,ie,je,ke
!      real :: res_ele
       real,shared :: sm_temp(*)

       ii = threadidx%x
       jj = blockidx%x
       kk = blockidx%y
       i = ib+ii-1
       j = jb+jj-1
       k = kb+kk-1

  !     index = dev_nim/2
!      index = blockDim%x/2
!      index_res = mod(blockDim%x, 2)
!      dev_parsum3(ii,jj,kk)=phi(i,j,k,m)
!      call syncthreads()
       sm_temp(ii) = 0.0
       if(i.ge.ib.and.i.le.ie)sm_temp(ii)=phi(i,j,k,m)
       call syncthreads()

       do while(index.ge.1)
       if(ii.le.index)then 
       sm_temp(ii) = sm_temp(ii) + sm_temp(index+ii)
!          res_ele = merge(sm_temp(index+ii+1), 0.0, ii.eq.index)
!          sm_temp(ii) = sm_temp(ii) + sm_temp(index+ii)
! to handle the case when the array lenghths are odd
!    1               + merge(res_ele, 0.0, index_res.gt.0 ) 
       endif
!      index_res = mod(index, 2)
       index = index/2
       call syncthreads()
       enddo

       if(threadidx%x.eq.1) then
!       dev_parsum2(jj,kk)=sm_temp(1)
       endif      

       end subroutine kernel_par_sum1_sm
        


      attributes(global) &
     & subroutine kernel_par_sum2_sm

       implicit none
       integer:: ii,jj,index,index_res
       real, shared :: sm_temp(*)
       real:: res_ele

       ii = threadidx%x
       jj = blockidx%x
!       sm_temp(ii)=dev_parsum2(ii,jj)
       call syncthreads()
   !    index = dev_njm/2
!      index = blockDim%x/2
!      index_res = mod(blockDim%x, 2)
       
       do while(index.ge.1)
       if (ii.le.index) then
       sm_temp(ii) = sm_temp(ii) + sm_temp(index+ii)
!          res_ele = merge(sm_temp(index+ii+1), 0.0, ii.eq.index)
!          sm_temp(ii) = sm_temp(ii) + sm_temp(index+ii)
! to handle the case when the array lenghths are odd
!    1               + merge(res_ele, 0.0, index_res.gt.0 ) 
       endif
!      index_res = mod(index, 2)
       index = index/2
       call syncthreads()
       enddo
       
       if(threadidx%x.eq.1) then

!       dev_parsum1(jj)=sm_temp(1)

       endif      
       endsubroutine kernel_par_sum2_sm 




      attributes(global)  &
     & subroutine kernel_par_sum3_sm(dsumm)

       implicit none
       integer:: ii,index,index_res
       real dsumm,res_ele
       real, shared :: sm_temp(*)

       ii = threadidx%x
!       sm_temp(ii)=dev_parsum1(ii)
       call syncthreads()
  !     index = dev_nkm/2
!      index = blockDim%x/2
!      index_res = mod(blockDim%x, 2)
       
       do while(index.ge.1)
       if (ii.le.index) then
       sm_temp(ii) = sm_temp(ii) + sm_temp(index+ii)
!          res_ele = merge(sm_temp(index+ii+1), 0.0, ii.eq.index)
!          sm_temp(ii) = sm_temp(ii) + sm_temp(index+ii)
! to handle the case when the array lenghths are odd
!    1               + merge(res_ele, 0.0, index_res.gt.0 ) 
       endif
!      index_res = mod(index, 2)
       index = index/2
       call syncthreads()
       enddo
       
       if(threadidx%x.eq.1) then
           dsumm = sm_temp(1)
       endif      
       endsubroutine kernel_par_sum3_sm 


      ! this assumes the data to be 4D but does only for on sub 3D array.
      end module kernels
      module dims
            integer::ni,nj,nk,nm
      end module dims
      module data_
            use cudafor
            
            implicit none
            
            real(8), allocatable,device,dimension(:,:,:)::dev_data3,dev_data3_2
            real(8), allocatable,device,dimension(:,:,:,:)::dev_data4
            real(8), allocatable, dimension(:,:,:)::data3
            real(8), allocatable, dimension(:,:,:,:)::data4
            real(8), device::reduction
            integer::ni,nj,nk,nm

!            allocate(dev_data3(1,1,1))
!            print *,"allocated"
      
      end module data_

      !This does the host and device data allocations.
      subroutine data_allocate(i,j,k,m)
            use data_

            implicit none
            integer,intent(in)::i,j,k,m
            integer::istat
!            ni = i
!            nj = j
!            nk = k
!            nm = m
            print *,"ni:",ni
            print *,"nj:",nj
            print *,"nk:",nk
            allocate(data3(ni,nj,nk),data4(ni,nj,nk,nm),stat=istat)
            print *,"Status:",istat
            allocate(dev_data3(ni,nj,nk),dev_data3_2(ni,nj,nk),dev_data4(ni,nj,nk,nm),stat=istat)
            print *,"Status:",istat
            print *,"Data Allocated"
      end subroutine data_allocate

      !initilialize the host side data that has to be reduced here.
      !For now I initialized it to 1.0
      subroutine data_initialize
            use data_
            implicit none
            data3 = 1.0
            data4 = 1.0
      end subroutine data_initialize

      !Transfers data from host to device
      subroutine data_transfer_h2d
            use data_
            implicit none
            dev_data3(:,:,:) = data3(:,:,:)
            dev_data4(:,:,:,:) = data4(:,:,:,:)
            dev_data3_2(:,:,:) = data3(:,:,:)
      end subroutine data_transfer_h2d
      subroutine deallocate_
            use data_
            implicit none
            deallocate(dev_data3)
            deallocate(data3)
            deallocate(dev_data3_2)
            deallocate(dev_data4)
            deallocate(data4)
      end subroutine deallocate_
      subroutine gpu_initialize

            use cudafor

            implicit none

            integer::istat, deviceused, idevice

            ! Initialize GPU
            istat = cudaSetDevice(idevice)

            ! cudaChooseDevice
            ! Tell me which GPU I use
            istat = cudaGetDevice(deviceused)
            print *, 'Device used', deviceused

      end subroutine gpu_initialize
      
      program main
            use kernels
            use data_
            use cudafor
            implicit none
            integer::tx,ty,tz,gx,gy,gz,istat,i
            type(dim3)::dimgrid,dimblock
            character(len=32)::args
            real(8)::sum_dot_gpu
            real(8), allocatable, device, dimension(:,:) :: dev_,DEV2
            i = command_argument_count() 
            if(i.lt.7) then
                  print *,"<ni><nj><nk><nm><tblockx><tblocky><tblockz>"
                  stop
            end if
            call getarg(1,args)
            read(args,'(I10)') ni
            call getarg(2,args)
            read(args,'(I10)') nj
            call getarg(3,args)
            read(args,'(I10)') nk
            
            call getarg(4,args)
            read(args,'(I10)') nm
            
            call getarg(5,args)
            read(args,'(I10)') tx
            call getarg(6,args)
            read(args,'(I10)') ty
            call getarg(7,args)
            read(args,'(I10)') tz
            call gpu_initialize
            call data_allocate(ni,nj,nk,nm)
            call data_initialize
            call data_transfer_h2d
            print *,'Performing reduction'
            dev_data3 = 1.0
!            tx = 8
!            ty = 8
!            tz = 2
!            gx = 0 
!            gy = 0
!            gz = 0
            dimblock = dim3(tx,ty,tz)
            print *,"ni:",ni
            print *,"nj:",nj
            print *,"nk:",nk
            print *,"gyr:",modulo(ni-2,ty)

            print *,"gxr:",modulo(ni-2,tx)
            print *,"gzr:",modulo(nk-2,tz)
            if(modulo(ni-2,ty).ne.0)then
                  gy = (ni-2)/ty +1
            else
                  gy = (ni-2)/ty
            end if
            if(modulo(nj-2,tx).ne.0)then
                  gx = (nj-2)/tx +1
            else
                  gx = (nj-2)/tx
            end if
            if(modulo(nk-2,tz).ne.0)then
                  gz = (nk-2)/tz +1
            else
                  gz = (nk-2)/tz
            end if
            dimgrid = dim3(gx,gy,1)
            print *,"gx:",gx
            print *,"gy:",gy
            print *,"gz:",gz
            do i=1,10
            reduction = 0.0
           call kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*8>>>(dev_data3 &
           & ,dev_data3_2,ni,nj,nk,2,2,2,nj-1,ni-1,nk-1,gz,reduction,tx*ty*tz)
            istat = cudathreadsynchronize()
            sum_dot_gpu = reduction
            
            print *, "Test Reduction:",sum_dot_gpu
            end do
            call deallocate_
      end program main
