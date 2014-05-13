      module afosr_cfd_c
      use ISO_C_BINDING
      real (C_DOUBLE) :: db
      real (C_FLOAT) :: fl
      integer (C_INT) :: in

      
      interface accel_alloc_c
      function accel_alloc_c(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: accel_alloc_c
      type (C_PTR), VALUE :: ptr
      integer (C_SIZE_T) :: size
      end function accel_alloc_c
      end interface

      interface accel_free_c
      function accel_free_c(ptr)
      use ISO_C_BINDING
      integer (C_INT) :: accel_free_c
      type (C_PTR), VALUE :: ptr
      end function accel_free_c
      end interface

      interface choose_accel
      function choose_accel_c(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: choose_accel
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function choose_accel_c
      end interface

      interface get_accel
      function get_accel_c(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: get_accel
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function get_accel_c
      end interface

      interface accel_validate_worksize_c
      function accel_validate_worksize_c(grid_x, grid_y, grid_z, block_x,&
      & block_y, block_z)
      use ISO_C_BINDING
      integer (C_INT) :: accel_validate_worksize
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      end function accel_validate_worksize_c
      end interface

      interface accel_dotprod_c
      function accel_dotprod_c(grid_x, grid_y, grid_z, block_x, block_y,&
      & block_z, data1, data2, size_x, size_y, size_z, start_x, start_y&
      &, start_z, end_x, end_y, end_z, reduction_var, a_type, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_dotprod_c
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      type (C_PTR), VALUE :: data1, data2
      integer (C_SIZE_T) :: size_x, size_y, size_z
      integer (C_SIZE_T) :: start_x, start_y, start_z
      integer (C_SIZE_T) :: end_x, end_y, end_z
      type (C_PTR), VALUE :: reduction_var
      integer (C_INT) :: a_type
      integer (C_INT) :: async
      end function accel_dotprod_c
      end interface

      interface accel_reduce_c
      function accel_reduce_c(grid_x, grid_y, grid_z, block_x, block_y,&
      & block_z, data, size_x, size_y, size_z, start_x, start_y&
      &, start_z, end_x, end_y, end_z, reduction_var, a_type, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_reduce_c
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      type (C_PTR), VALUE :: data
      integer (C_SIZE_T) :: size_x, size_y, size_z
      integer (C_SIZE_T) :: start_x, start_y, start_z
      integer (C_SIZE_T) :: end_x, end_y, end_z
      type (C_PTR), VALUE :: reduction_var
      integer (C_INT) :: a_type
      integer (C_INT) :: async
      end function accel_reduce_c
      end interface

      interface accel_copy_h2d_c
      function accel_copy_h2d_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_h2d_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function accel_copy_h2d_c
      end interface

! we need an interface and special versions for each rank of source
!  since Fortran apparently can't just treat src like a void *
!      interface accel_copy_h2d
!      function accel_copy_h2d_1d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: accel_copy_h2d_1d
!      class(*), TARGET :: dst
!      class(*), dimension(:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function accel_copy_h2d_1d
!      function accel_copy_h2d_2d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: accel_copy_h2d_2d
!      class(*), TARGET :: dst
!      class(*), dimension(:,:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function accel_copy_h2d_2d
!      function accel_copy_h2d_3d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: accel_copy_h2d_3d
!      class(*), TARGET :: dst
!      class(*), dimension(:,:,:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function accel_copy_h2d_3d
!      function accel_copy_h2d_4d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: accel_copy_h2d_4d
!      class(*), TARGET :: dst
!      class(*), dimension(:,:,:,:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function accel_copy_h2d_4d
!      end interface

      interface accel_copy_d2h_c
      function accel_copy_d2h_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_d2h_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function accel_copy_d2h_c
      end interface

      interface accel_copy_d2d_c
      function accel_copy_d2d_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_d2d_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function accel_copy_d2d_c
      end interface

#ifdef WITH_TIMERS

      interface accel_timers_init
      function accel_timers_init_c()
      use ISO_C_BINDING
      integer (C_INT) :: accel_timers_init_c
      end function accel_timers_init_c
      end interface

      interface accel_timers_flush
      function accel_timers_flush_c()
      use ISO_C_BINDING
      integer (C_INT) :: accel_timers_flush_c
      end function accel_timers_flush_c
      end interface

      interface accel_timers_finish
      function accel_timers_finish_c()
      use ISO_C_BINDING
      integer (C_INT) :: accel_timers_finish_c
      end function accel_timers_finish_c
      end interface
      
#endif

      contains
      function accel_alloc(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: accel_alloc
      class(*), target :: ptr
      integer :: size
      select type(ptr)
      type is (integer(kind=C_INT))
           accel_alloc = accel_alloc_c(C_LOC(ptr), &
            int(C_SIZEOF(in)*size, kind=C_SIZE_T))
      type is (real(kind=C_FLOAT))
           accel_alloc = accel_alloc_c(C_LOC(ptr), &
            int(C_SIZEOF(fl)*size, kind=C_SIZE_T))
      type is (real(kind=C_DOUBLE))
           accel_alloc = accel_alloc_c(C_LOC(ptr), &
            int(C_SIZEOF(db)*size, kind=C_SIZE_T))
      class default
           ! do something
      end select
      end function accel_alloc

      function accel_free(ptr)
      use ISO_C_BINDING
      integer (C_INT) :: accel_free
      class(*), TARGET :: ptr
      select type (ptr)
      type is (integer(kind=C_INT))
           accel_free = accel_free_c(C_LOC(ptr))
      type is (real(kind=C_FLOAT))
           accel_free = accel_free_c(C_LOC(ptr))
      type is (real(kind=C_DOUBLE))
           accel_free = accel_free_c(C_LOC(ptr))
      class default
           ! do something
      end select
      end function accel_free

      function accel_validate_worksize(grid, block)
      use ISO_C_BINDING
      integer (C_INT) :: accel_validate_worksize
      integer, dimension(3) :: grid, block
      accel_validate_worksize = accel_validate_worksize_c( &
       int(grid(1),kind=C_SIZE_T), int(grid(2),kind=C_SIZE_T), &
       int(grid(3),kind=C_SIZE_T), int(block(1),kind=C_SIZE_T), &
       int(block(2),kind=C_SIZE_T), int(block(3),kind=C_SIZE_T))
      end function accel_validate_worksize
      
      function accel_dotprod(grid, block, data1, data2, sz, start_idx&
      &, end_idx, reduction_var, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_dotprod
      integer, dimension(3) :: grid
      integer, dimension(3) :: block
      class(*), TARGET :: data1, data2, reduction_var
      integer, dimension(3) :: sz
      integer, dimension(3) :: start_idx
      integer, dimension(3) :: end_idx
      integer (C_INT) :: async
      select type(reduction_var)
      type is(integer(kind=C_INT))
! since Fortran2003 doesn't appear to let us specify that data1, data2,
!  and reduction_var MUST be the same polymorphic type, we nest selects
       select type (data1)
       type is (integer(kind=C_INT))
        select type(data2)
        type is (integer(kind=C_INT))
           accel_dotprod = accel_dotprod_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data1), C_LOC(data2), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), 4, &
            async)
        end select
       end select
      type is(real(kind=C_FLOAT))
       select type (data1)
       type is (real(kind=C_FLOAT))
        select type (data2)
        type is (real(kind=C_FLOAT))
           accel_dotprod = accel_dotprod_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data1), C_LOC(data2), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), 1, &
            async)
        end select
       end select
      type is(real(kind=C_DOUBLE))
       select type (data1)
       type is (real(kind=C_DOUBLE))
        select type (data2)
        type is (real(kind=C_DOUBLE))
           accel_dotprod = accel_dotprod_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data1), C_LOC(data2), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), 0, &
            async)
        end select
       end select
      class default
            !do seomthing
      end select
      end function accel_dotprod
      
      function accel_reduce(grid, block, data, sz, start_idx&
      &, end_idx, reduction_var, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_reduce
      integer, dimension(3) :: grid
      integer, dimension(3) :: block
      class(*), TARGET :: data, reduction_var
      integer, dimension(3) :: sz
      integer, dimension(3) :: start_idx
      integer, dimension(3) :: end_idx
      integer (C_INT) :: async
      select type(reduction_var)
      type is(integer(kind=C_INT))
! since Fortran2003 doesn't appear to let us specify that data1, data2,
!  and reduction_var MUST be the same polymorphic type, we nest selects
       select type (data)
       type is (integer(kind=C_INT))
           accel_reduce = accel_reduce_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), 4, &
            async)
       end select
      type is(real(kind=C_FLOAT))
       select type (data)
       type is (real(kind=C_FLOAT))
           accel_reduce = accel_reduce_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), 1, &
            async)
       end select
      type is(real(kind=C_DOUBLE))
       select type (data)
       type is (real(kind=C_DOUBLE))
           accel_reduce = accel_reduce_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), 0, &
            async)
       end select
      class default
            !do seomthing
      end select
      end function accel_reduce


      function accel_copy_h2d(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_h2d
      class(*), TARGET :: dst
      type(*), dimension(*), TARGET :: src
      integer :: size
      integer :: async
      select type (dst)
       type is (integer(kind=C_INT))
            accel_copy_h2d = accel_copy_h2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(in),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_FLOAT))
            accel_copy_h2d = accel_copy_h2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(fl),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_DOUBLE))
            accel_copy_h2d = accel_copy_h2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(db),kind=C_SIZE_T),int(async,kind=C_INT))
       class default
            ! do something
      end select
      end function accel_copy_h2d

      function accel_copy_d2h(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_d2h
      class(*), TARGET :: src 
      type(*), dimension(*), TARGET :: dst
      integer :: size
      integer :: async
      select type (src)
       type is (integer(kind=C_INT))
            accel_copy_d2h = accel_copy_d2h_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(in),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_FLOAT))
            accel_copy_d2h = accel_copy_d2h_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(fl),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_DOUBLE))
            accel_copy_d2h = accel_copy_d2h_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(db),kind=C_SIZE_T),int(async,kind=C_INT))
       class default
            ! do something
      end select
      end function accel_copy_d2h

      function accel_copy_d2d(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_d2d
      class(*), TARGET :: dst, src
      integer :: size
      integer :: async
      select type (dst)
       type is (integer(kind=C_INT))
       select type (src)
        type is (integer(kind=C_INT))
            accel_copy_d2d = accel_copy_d2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(in),kind=C_SIZE_T),int(async,kind=C_INT))
        end select
       type is (real(kind=C_FLOAT))
       select type (src)
        type is (real(kind=C_FLOAT))
            accel_copy_d2d = accel_copy_d2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(fl),kind=C_SIZE_T),int(async,kind=C_INT))
        end select
       type is (real(kind=C_DOUBLE))
       select type (src)
        type is (real(kind=C_DOUBLE))
            accel_copy_d2d = accel_copy_d2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(db),kind=C_SIZE_T),int(async,kind=C_INT))
        end select
       class default
            ! do something
      end select
      end function accel_copy_d2d

      end module afosr_cfd_c
