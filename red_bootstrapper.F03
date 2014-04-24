#include "afosr_cfd_fortran.F03"
!      include 'afosr_cfd_fortran.f03'
      
      module dims
            integer::ni,nj,nk,nm
      end module dims
      module data_
            use ISO_C_BINDING
            implicit none
            
            type (C_PTR), TARGET :: dev_data3, dev_data3_2, dev_data4
            real(C_DOUBLE), TARGET, allocatable, dimension(:,:,:)::data3
            real(C_DOUBLE), TARGET, allocatable, dimension(:,:,:,:)::data4
            type (C_PTR), TARGET :: reduction
            integer::ni,nj,nk,nm

!            allocate(dev_data3(1,1,1))
!            print *,"allocated"
      
      end module data_

      !This does the host and device data allocations.
      subroutine data_allocate(i,j,k,m)
            use data_
            use ISO_C_BINDING
            use afosr_cfd_c

            implicit none
            integer,intent(in)::i,j,k,m
            integer::istat
            real (C_DOUBLE) :: double
!            ni = i
!            nj = j
!            nk = k
!            nm = m
            print *,"ni:",ni
            print *,"nj:",nj
            print *,"nk:",nk
            allocate(data3(ni,nj,nk),data4(ni,nj,nk,nm),stat=istat)
            print *,"Status:",istat
!            allocate(dev_data3(ni,nj,nk),dev_data3_2(ni,nj,nk),dev_data4(ni,nj,nk,nm),stat=istat)
            istat = accel_alloc(C_LOC(dev_data3), ni*nj*nk*C_SIZEOF(double))
            print *,"Status:",istat
            istat = accel_alloc(C_LOC(dev_data4), ni*nj*nk*nm*C_SIZEOF(double))
            print *,"Status:",istat
            istat = accel_alloc(C_LOC(dev_data3_2), ni*nj*nk*C_SIZEOF(double))
            print *,"Status:",istat
            istat = accel_alloc(C_LOC(reduction), C_SIZEOF(double))
            print *,"Status:",istat
            print *,"Data Allocated"
      end subroutine data_allocate

      !initilialize the host side data that has to be reduced here.
      !For now I initialized it to 1.0
      subroutine data_initialize
            use data_
            implicit none
            integer :: i, j, k
            data3 = 1.0
            data4 = 1.0
            do i = 1, ni
              do j = 1, nj
                do k = 1, nk
                  if (i.eq.1.OR.j.eq.1.OR.k.eq.1) then
                    data3(i,j,k)=0.0
                  end if
                  if (i.eq.ni.OR.j.eq.nj.OR.k.eq.nk) then
                    data3(i,j,k)=0.0
                  end if 
                end do
              end do
            end do
      end subroutine data_initialize

      !Transfers data from host to device
      subroutine data_transfer_h2d
            use data_
            use ISO_C_BINDING
            use afosr_cfd_c
            implicit none
            integer::istat
            real (C_DOUBLE) :: double
!            dev_data3(:,:,:) = data3(:,:,:)
            istat = accel_copy_h2d(dev_data3, C_LOC(data3), ni*nj*nk*C_SIZEOF(double), 1)
            istat = accel_copy_h2d(dev_data4, C_LOC(data4), ni*nj*nk*nm*C_SIZEOF(double), 1)
            istat = accel_copy_d2d(dev_data3_2, dev_data3, ni*nj*nk*C_SIZEOF(double), 1)
!            dev_data4(:,:,:,:) = data4(:,:,:,:)
!            dev_data3_2(:,:,:) = data3(:,:,:)
      end subroutine data_transfer_h2d
      subroutine deallocate_
            use data_
            use ISO_C_BINDING
            use afosr_cfd_c
            implicit none
            integer::istat
!            deallocate(dev_data3)

            istat = accel_free(dev_data3)
            deallocate(data3)
!            deallocate(dev_data3_2)
            istat = accel_free(dev_data3_2)
!            deallocate(dev_data4)
            istat = accel_free(dev_data4)
            deallocate(data4)
      end subroutine deallocate_
      subroutine gpu_initialize
            use ISO_C_BINDING
            use afosr_cfd_c
            implicit none

            integer::istat, deviceused
            integer (C_INT) :: idevice, mode
            idevice = -1
            ! Initialize GPU
!            istat = cudaSetDevice(idevice)
            istat = choose_accel(idevice, 0)

            ! cudaChooseDevice
            ! Tell me which GPU I use
!            istat = cudaGetDevice(deviceused)
            istat = get_accel(deviceused, mode)
            print *, 'Device used', deviceused

      end subroutine gpu_initialize
      
      program main
            use data_
            use ISO_C_BINDING
            use afosr_cfd_c
            implicit none
            integer :: tx, ty, tz, gx, gy, gz
            integer::istat,i
            character(len=32)::args
            real(C_DOUBLE), TARGET::sum_dot_gpu, zero
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

#ifdef WITH_TIMERS
            istat = accel_timers_init()
#endif
            call gpu_initialize
            call data_allocate(ni,nj,nk,nm)
            call data_initialize
            call data_transfer_h2d
            print *,'Performing reduction'
!            tx = 8
!            ty = 8
!            tz = 2
!            gx = 0 
!            gy = 0
!            gz = 0
!            dimblock = dim3(tx,ty,tz)
            print *,"ni:",ni
            print *,"nj:",nj
            print *,"nk:",nk
            print *,"gyr:",modulo(nj-2,ty)

            print *,"gxr:",modulo(ni-2,tx)
            print *,"gzr:",modulo(nk-2,tz)
            if(modulo(nj-2,ty).ne.0)then
                  gy = (nj-2)/ty +1
            else
                  gy = (nj-2)/ty
            end if
            if(modulo(ni-2,tx).ne.0)then
                  gx = (ni-2)/tx +1
            else
                  gx = (ni-2)/tx
            end if
            if(modulo(nk-2,tz).ne.0)then
                  gz = (nk-2)/tz +1
            else
                  gz = (nk-2)/tz
            end if
!            dimgrid = dim3(gx,gy,1)
            print *,"gx:",gx
            print *,"gy:",gy
            print *,"gz:",gz
            do i=1,10
!            reduction = 0.0
           zero = 0
           istat = accel_copy_h2d(reduction, C_LOC(zero), C_SIZEOF(zero), 1)
           istat = accel_dotprod(int(gx,kind=C_SIZE_T),int(gy,kind=C_SIZE_T),&
            & int(gz*tz,kind=C_SIZE_T),int(tx,kind=C_SIZE_T),int(ty,kind=C_SIZE_T),&
            & int(1,kind=C_SIZE_T),dev_data3, dev_data3_2, int(ni,kind=C_SIZE_T),&
            & int(nj,kind=C_SIZE_T),int(nk,kind=C_SIZE_T),int(2,kind=C_SIZE_T),&
            & int(2,kind=C_SIZE_T),int(2,kind=C_SIZE_T),int(ni-1,kind=C_SIZE_T),&
            & int(nj-1,kind=C_SIZE_T),int(nk-1,kind=C_SIZE_T),reduction&
            &, int(0,kind=C_INT))
!           istat = accel_dotprod(int(gx,kind=C_SIZE_T),int(gy,kind=C_SIZE_T),&
!            & int(gz,kind=C_SIZE_T),int(tx,kind=C_SIZE_T),int(ty,kind=C_SIZE_T),&
!            & int(tz,kind=C_SIZE_T),dev_data3, dev_data3_2, int(ni,kind=C_SIZE_T),&
!            & int(nj,kind=C_SIZE_T),int(nk,kind=C_SIZE_T),int(2,kind=C_SIZE_T),&
!            & int(2,kind=C_SIZE_T),int(2,kind=C_SIZE_T),int(nj-1,kind=C_SIZE_T),&
!            & int(ni-1,kind=C_SIZE_T),int(nk-1,kind=C_SIZE_T),reduction&
!            &, int(0,kind=C_INT))
!           call kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*8>>>(dev_data3 &
!           & ,dev_data3_2,ni,nj,nk,2,2,2,nj-1,ni-1,nk-1,gz,reduction,tx*ty*tz)
!            istat = cudathreadsynchronize()
!            sum_dot_gpu = reduction
           istat = accel_copy_d2h(C_LOC(sum_dot_gpu), reduction, C_SIZEOF(sum_dot_gpu), 0)
            
            print *, "Test Reduction:",sum_dot_gpu
            end do
            call deallocate_
#ifdef WITH_TIMERS
           istat = accel_timers_finish()
#endif
      end program main
