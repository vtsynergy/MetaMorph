include "metamorph_fortran_header.F03"
!#include <sys/time.h>
!      include 'metamorph_fortran.f03'
      
      module data_
            use ISO_C_BINDING
            implicit none
           
! Device variables must all be pointers to their appropriate C_TYPE
! Host buffers which exchange with device buffers must all be 
!  non-scalar, TARGET, allocatables of appropriate C_TYPE 
            real(C_DOUBLE), POINTER :: dev_data3_db, dev_data3_2_db, dev_data4_db
            real(C_DOUBLE), TARGET, allocatable, dimension(:,:,:)::data3_db
            real(C_DOUBLE), TARGET, allocatable, dimension(:,:,:,:)::data4_db
            real(C_DOUBLE), POINTER :: reduction_db
            
            real(C_FLOAT), POINTER :: dev_data3_fl, dev_data3_2_fl, dev_data4_fl
            real(C_FLOAT), TARGET, allocatable, dimension(:,:,:)::data3_fl
            real(C_FLOAT), TARGET, allocatable, dimension(:,:,:,:)::data4_fl
            real(C_FLOAT), POINTER :: reduction_fl
            
            integer(C_INT), POINTER :: dev_data3_in, dev_data3_2_in, dev_data4_in
            integer(C_INT), TARGET, allocatable, dimension(:,:,:)::data3_in
            integer(C_INT), TARGET, allocatable, dimension(:,:,:,:)::data4_in
            integer(C_INT), POINTER :: reduction_in
            integer::ni,nj,nk,nm,g_type

!            allocate(dev_data3(1,1,1))
!            print *,"allocated"
      
      end module data_

      !This does the host and device data allocations.
      subroutine data_allocate(i,j,k,m)
            use data_
            use ISO_C_BINDING
            use metamorph_c

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
            if (g_type.EQ.0) then
            allocate(data3_db(ni,nj,nk),data4_db(ni,nj,nk,nm),stat=istat)
            print *,"Status:",istat
!            allocate(dev_data3(ni,nj,nk),dev_data3_2(ni,nj,nk),dev_data4(ni,nj,nk,nm),stat=istat)
            istat = meta_alloc(dev_data3_db, ni*nj*nk)
            print *,"Status:",istat
            istat = meta_alloc(dev_data4_db, ni*nj*nk*nm)
            print *,"Status:",istat
            istat = meta_alloc(dev_data3_2_db, ni*nj*nk)
            print *,"Status:",istat
            istat = meta_alloc(reduction_db, 1)
            print *,"Status:",istat
            endif
            if (g_type.EQ.1) then
            allocate(data3_fl(ni,nj,nk),data4_fl(ni,nj,nk,nm),stat=istat)
            print *,"Status:",istat
!            allocate(dev_data3(ni,nj,nk),dev_data3_2(ni,nj,nk),dev_data4(ni,nj,nk,nm),stat=istat)
            istat = meta_alloc(dev_data3_fl, ni*nj*nk)
            print *,"Status:",istat
            istat = meta_alloc(dev_data4_fl, ni*nj*nk*nm)
            print *,"Status:",istat
            istat = meta_alloc(dev_data3_2_fl, ni*nj*nk)
            print *,"Status:",istat
            istat = meta_alloc(reduction_fl, 1)
            print *,"Status:",istat
            endif
            if (g_type.EQ.4) then
            allocate(data3_in(ni,nj,nk),data4_in(ni,nj,nk,nm),stat=istat)
            print *,"Status:",istat
!            allocate(dev_data3(ni,nj,nk),dev_data3_2(ni,nj,nk),dev_data4(ni,nj,nk,nm),stat=istat)
            istat = meta_alloc(dev_data3_in, ni*nj*nk)
            print *,"Status:",istat
            istat = meta_alloc(dev_data4_in, ni*nj*nk*nm)
            print *,"Status:",istat
            istat = meta_alloc(dev_data3_2_in, ni*nj*nk)
            print *,"Status:",istat
            istat = meta_alloc(reduction_in, 1)
            print *,"Status:",istat
            endif
            print *,"Data Allocated"
      end subroutine data_allocate

      !initilialize the host side data that has to be reduced here.
      !For now I initialized it to 1.0
      subroutine data_initialize
            use data_
            implicit none
            integer :: i, j, k
            if (g_type.EQ.0) then
            data3_db = 1.0
            data4_db = 1.0
            do i = 1, ni
              do j = 1, nj
                do k = 1, nk
                  if (i.eq.1.OR.j.eq.1.OR.k.eq.1) then
                    data3_db(i,j,k)=0.0
                  end if
                  if (i.eq.ni.OR.j.eq.nj.OR.k.eq.nk) then
                    data3_db(i,j,k)=0.0
                  end if 
                end do
              end do
            end do
            endif
            if (g_type.EQ.1) then
            data3_fl = 1.0
            data4_fl = 1.0
            do i = 1, ni
              do j = 1, nj
                do k = 1, nk
                  if (i.eq.1.OR.j.eq.1.OR.k.eq.1) then
                    data3_fl(i,j,k)=0.0
                  end if
                  if (i.eq.ni.OR.j.eq.nj.OR.k.eq.nk) then
                    data3_fl(i,j,k)=0.0
                  end if 
                end do
              end do
            end do
            endif
            if (g_type.EQ.4) then
            data3_in = 1
            data4_in = 1
            do i = 1, ni
              do j = 1, nj
                do k = 1, nk
                  if (i.eq.1.OR.j.eq.1.OR.k.eq.1) then
                    data3_in(i,j,k)=0
                  end if
                  if (i.eq.ni.OR.j.eq.nj.OR.k.eq.nk) then
                    data3_in(i,j,k)=0
                  end if 
                end do
              end do
            end do
            endif
      end subroutine data_initialize

      !Transfers data from host to device
      subroutine data_transfer_h2d
            use data_
            use ISO_C_BINDING
            use metamorph_c
            implicit none
            integer::istat
            integer::iter
            integer:: elapsed
            external:: elapsed
            real(C_DOUBLE):: stime, etime

!            dev_data3(:,:,:) = data3(:,:,:)
!TODO add timing loops
            if (g_type.EQ.0) then
            istat = meta_copy_h2d(dev_data3_db, data3_db, ni*nj*nk, 1)
            istat = meta_copy_h2d(dev_data4_db, data4_db, ni*nj*nk*nm, 1)
            istat = meta_copy_d2d(dev_data3_2_db, dev_data3_db, ni*nj*nk, 1)
            endif
            if (g_type.EQ.1) then
            istat = elapsed(stime)
            do iter = 0, 999
            istat = meta_copy_h2d(dev_data3_fl, data3_fl, ni*nj*nk, 0)
            enddo
            istat = elapsed(etime)
            print *, "H2D time: ", (etime-stime)/(1000.0)
            istat = meta_copy_h2d(dev_data4_fl, data4_fl, ni*nj*nk*nm, 0)
            istat = elapsed(stime)
            do iter = 0, 999
            istat = meta_copy_d2d(dev_data3_2_fl, dev_data3_fl, ni*nj*nk, 0)
            enddo
            istat = elapsed(etime)
            print *, "D2D time: ", (etime-stime)/(1000.0)
            endif
            if (g_type.EQ.4) then
            istat = meta_copy_h2d(dev_data3_in, data3_in, ni*nj*nk, 1)
            istat = meta_copy_h2d(dev_data4_in, data4_in, ni*nj*nk*nm, 1)
            istat = meta_copy_d2d(dev_data3_2_in, dev_data3_in, ni*nj*nk, 1)
            endif
!            dev_data4(:,:,:,:) = data4(:,:,:,:)
!            dev_data3_2(:,:,:) = data3(:,:,:)
      end subroutine data_transfer_h2d
      subroutine deallocate_
            use data_
            use ISO_C_BINDING
            use metamorph_c
            implicit none
            integer::istat
!            deallocate(dev_data3)
            if (g_type.EQ.0) then
            istat = meta_free(dev_data3_db)
            deallocate(data3_db)
!            deallocate(dev_data3_2)
            istat = meta_free(dev_data3_2_db)
!            deallocate(dev_data4)
            istat = meta_free(dev_data4_db)
            deallocate(data4_db)
            endif
            if (g_type.EQ.1) then
            istat = meta_free(dev_data3_fl)
            deallocate(data3_fl)
!            deallocate(dev_data3_2)
            istat = meta_free(dev_data3_2_fl)
!            deallocate(dev_data4)
            istat = meta_free(dev_data4_fl)
            deallocate(data4_fl)
            endif
            if (g_type.EQ.4) then
            istat = meta_free(dev_data3_in)
            deallocate(data3_in)
!            deallocate(dev_data3_2)
            istat = meta_free(dev_data3_2_in)
!            deallocate(dev_data4)
            istat = meta_free(dev_data4_in)
            deallocate(data4_in)
            endif
      end subroutine deallocate_
      subroutine gpu_initialize
            use ISO_C_BINDING
            use metamorph_c
            implicit none

            integer::istat, deviceused
            integer (C_INT) :: idevice, mode
            idevice = -1
            ! Initialize GPU
!            istat = cudaSetDevice(idevice)
            istat = meta_set_acc(idevice, 0)

            ! cudaChooseDevice
            ! Tell me which GPU I use
!            istat = cudaGetDevice(deviceused)
            istat = meta_get_acc(deviceused, mode)
            print *, 'Device used', deviceused

      end subroutine gpu_initialize
      
      program main
            use data_
            use ISO_C_BINDING
            use metamorph_c
            implicit none
            integer :: tx, ty, tz, gx, gy, gz
            integer::istat,i, iter
            integer:: elapsed
            external:: elapsed
            character(len=32)::args
            real(C_DOUBLE), TARGET, dimension(1)::sum_dot_gpu_db, zero_db
            real(C_FLOAT), TARGET, dimension(1)::sum_dot_gpu_fl, zero_fl
            integer(C_INT), TARGET, dimension(1)::sum_dot_gpu_in, zero_in
            real(C_DOUBLE):: etime, stime
            i = command_argument_count() 
            if(i.lt.8) then
                  print *,"<ni><nj><nk><nm><tblockx><tblocky><tblockz>t"
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

            call getarg(8,args)
            read(args,'(I10)') g_type


#ifdef WITH_TIMERS
            istat = meta_timers_init()
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
            if(modulo(nj,ty).ne.0)then
                  gy = (nj)/ty +1
            else
                  gy = (nj)/ty
            end if
            if(modulo(ni,tx).ne.0)then
                  gx = (ni)/tx +1
            else
                  gx = (ni)/tx
            end if
            if(modulo(nk,tz).ne.0)then
                  gz = (nk)/tz +1
            else
                  gz = (nk)/tz
            end if
!            dimgrid = dim3(gx,gy,1)
            print *,"gx:",gx
            print *,"gy:",gy
            print *,"gz:",gz
!            do i=1,10
!            reduction = 0.0
           zero_db = 0.0
           zero_fl = 0.0
           zero_in = 0.0
           if (g_type.EQ.0) then
           istat = meta_copy_h2d(reduction_db, zero_db, 1, 1)
!TODO add timing loop
           istat = meta_dotprod((/gx, gy, gz*tz/),(/tx, ty, 1/), &
            dev_data3_db, dev_data3_2_db, (/ni, nj, nk/), (/1, 1, 1/), &
            (/ni, nj, nk/), reduction_db, 0)
!           istat = meta_dotprod(int(gx,kind=C_SIZE_T),int(gy,kind=C_SIZE_T),&
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
           istat = meta_copy_d2h(sum_dot_gpu_db, reduction_db, 1, 0)
            
            print *, "Test Reduction:",sum_dot_gpu_db
           endif
           if (g_type.EQ.1) then
           istat = meta_copy_h2d(reduction_fl, zero_fl, 1, 1)
           istat = elapsed(stime)
           do iter = 0, 999
           istat = meta_dotprod((/gx, gy, gz*tz/),(/tx, ty, 1/), &
            dev_data3_fl, dev_data3_2_fl, (/ni, nj, nk/), (/1, 1, 1/), &
            (/ni, nj, nk/), reduction_fl, 0)
           enddo
           istat = elapsed(etime)
            print *, "Kern time: ", (etime-stime)/(1000.0)
!           istat = meta_dotprod(int(gx,kind=C_SIZE_T),int(gy,kind=C_SIZE_T),&
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
           istat = meta_copy_d2h(sum_dot_gpu_fl, reduction_fl, 1, 0)
            
            print *, "Test Reduction:",sum_dot_gpu_fl
           istat = elapsed(stime)
           do iter = 0, 999
           istat = meta_copy_d2h(data3_fl, dev_data3_fl, ni*nj*nk, 0)
           enddo
           istat = elapsed(etime)
           print *, "D2H time: ", (etime-stime)/(1000.0)
           endif
           if (g_type.EQ.4) then
           istat = meta_copy_h2d(reduction_in, zero_in, 1, 1)
           istat = meta_dotprod((/gx, gy, gz*tz/),(/tx, ty, 1/), &
            dev_data3_in, dev_data3_2_in, (/ni, nj, nk/), (/2, 2, 2/), &
            (/ni-1, nj-1, nk-1/), reduction_in, 0)
!           istat = meta_dotprod(int(gx,kind=C_SIZE_T),int(gy,kind=C_SIZE_T),&
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
           istat = meta_copy_d2h(sum_dot_gpu_in, reduction_in, 1, 0)
            
            print *, "Test Reduction:",sum_dot_gpu_in
           endif
!            end do
            call deallocate_
#ifdef WITH_TIMERS
           istat = meta_timers_finish()
#endif
      end program main
