!> @file
      module metamorph_c
      use ISO_C_BINDING
      real (C_DOUBLE) :: db
      real (C_FLOAT) :: fl
      integer (C_INT) :: in

      
      interface meta_alloc_c
      function meta_alloc_c(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: meta_alloc_c
      type (C_PTR) :: ptr
      integer (C_SIZE_T) :: size
      end function meta_alloc_c
      end interface

      interface meta_free_c
      function meta_free_c(ptr)
      use ISO_C_BINDING
      integer (C_INT) :: meta_free_c
      type (C_PTR), VALUE :: ptr
      end function meta_free_c
      end interface

      interface meta_set_acc
      function meta_set_acc_c(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: meta_set_acc
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function meta_set_acc_c
      end interface

      interface meta_get_acc
      function meta_get_acc_c(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: meta_get_acc
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function meta_get_acc_c
      end interface

      interface meta_validate_worksize_c
      function meta_validate_worksize_c(grid_x, grid_y, grid_z, block_x,&
      & block_y, block_z)
      use ISO_C_BINDING
      integer (C_INT) :: meta_validate_worksize
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      end function meta_validate_worksize_c
      end interface

      interface meta_dotprod_c
      function meta_dotprod_c(grid_x, grid_y, grid_z, block_x, block_y,&
      & block_z, data1, data2, size_x, size_y, size_z, start_x, start_y&
      &, start_z, end_x, end_y, end_z, reduction_var, a_type, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_dotprod_c
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      type (C_PTR), VALUE :: data1, data2
      integer (C_SIZE_T) :: size_x, size_y, size_z
      integer (C_SIZE_T) :: start_x, start_y, start_z
      integer (C_SIZE_T) :: end_x, end_y, end_z
      type (C_PTR), VALUE :: reduction_var
      integer (C_INT), VALUE :: a_type
      integer (C_INT) :: async
      end function meta_dotprod_c
      end interface

      interface meta_reduce_c
      function meta_reduce_c(grid_x, grid_y, grid_z, block_x, block_y,&
      & block_z, data, size_x, size_y, size_z, start_x, start_y&
      &, start_z, end_x, end_y, end_z, reduction_var, a_type, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_reduce_c
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      type (C_PTR), VALUE :: data
      integer (C_SIZE_T) :: size_x, size_y, size_z
      integer (C_SIZE_T) :: start_x, start_y, start_z
      integer (C_SIZE_T) :: end_x, end_y, end_z
      type (C_PTR), VALUE :: reduction_var
      integer (C_INT), VALUE :: a_type
      integer (C_INT) :: async
      end function meta_reduce_c
      end interface

      interface meta_copy_h2d_c
      function meta_copy_h2d_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_h2d_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function meta_copy_h2d_c
      end interface

! we need an interface and special versions for each rank of source
!  since Fortran apparently can't just treat src like a void *
!      interface meta_copy_h2d
!      function meta_copy_h2d_1d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: meta_copy_h2d_1d
!      class(*), TARGET :: dst
!      class(*), dimension(:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function meta_copy_h2d_1d
!      function meta_copy_h2d_2d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: meta_copy_h2d_2d
!      class(*), TARGET :: dst
!      class(*), dimension(:,:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function meta_copy_h2d_2d
!      function meta_copy_h2d_3d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: meta_copy_h2d_3d
!      class(*), TARGET :: dst
!      class(*), dimension(:,:,:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function meta_copy_h2d_3d
!      function meta_copy_h2d_4d(dst, src, size, async)
!      use ISO_C_BINDING
!      integer (C_INT) :: meta_copy_h2d_4d
!      class(*), TARGET :: dst
!      class(*), dimension(:,:,:,:) :: src
!      integer :: size
!      integer (C_INT) :: async
!      end function meta_copy_h2d_4d
!      end interface

      interface meta_copy_d2h_c
      function meta_copy_d2h_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_d2h_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function meta_copy_d2h_c
      end interface

      interface meta_copy_d2d_c
      function meta_copy_d2d_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_d2d_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function meta_copy_d2d_c
      end interface

      interface meta_alloc
        module procedure meta_alloc_db, meta_alloc_fl, meta_alloc_in
      end interface meta_alloc

      contains

      function meta_alloc_db(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: meta_alloc_db
      real(C_DOUBLE), POINTER, intent(inout) :: ptr
      integer :: size
      type(C_PTR) :: cptr
      meta_alloc_db = meta_alloc_c(cptr, &
       int(C_SIZEOF(db)*size, kind=C_SIZE_T))
      call C_F_POINTER(cptr, ptr)
      end function meta_alloc_db

      function meta_alloc_fl(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: meta_alloc_fl
      real(C_FLOAT), POINTER, intent(inout) :: ptr
      integer :: size
      type(C_PTR) :: cptr
      meta_alloc_fl = meta_alloc_c(cptr, &
       int(C_SIZEOF(fl)*size, kind=C_SIZE_T))
      call C_F_POINTER(cptr, ptr) 
      end function meta_alloc_fl

      function meta_alloc_in(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: meta_alloc_in
      integer(C_INT), POINTER, intent(inout) :: ptr
      integer :: size
      type(C_PTR) :: cptr
      meta_alloc_in = meta_alloc_c(cptr, &
       int(C_SIZEOF(in)*size, kind=C_SIZE_T))
      call C_F_POINTER(cptr, ptr)
      end function meta_alloc_in
!      function meta_alloc(ptr, size)
!      use ISO_C_BINDING
!      integer (C_INT) :: meta_alloc
!      class(*), TARGET :: ptr
!      integer :: size
!      select type(ptr)
!      type is (integer(kind=C_INT))
!           meta_alloc = meta_alloc_c(C_LOC(ptr), &
!            int(C_SIZEOF(in)*size, kind=C_SIZE_T))
!      type is (real(kind=C_FLOAT))
!           meta_alloc = meta_alloc_c(C_LOC(ptr), &
!            int(C_SIZEOF(fl)*size, kind=C_SIZE_T))
!      type is (real(kind=C_DOUBLE))
!           meta_alloc = meta_alloc_c(C_LOC(ptr), &
!            int(C_SIZEOF(db)*size, kind=C_SIZE_T))
!      class default
           ! do something
!      end select
!      end function meta_alloc

      function meta_free(ptr)
      use ISO_C_BINDING
      integer (C_INT) :: meta_free
      class(*), TARGET :: ptr
      select type (ptr)
      type is (integer(kind=C_INT))
           meta_free = meta_free_c(C_LOC(ptr))
      type is (real(kind=C_FLOAT))
           meta_free = meta_free_c(C_LOC(ptr))
      type is (real(kind=C_DOUBLE))
           meta_free = meta_free_c(C_LOC(ptr))
      class default
           ! do something
      end select
      end function meta_free

      function meta_validate_worksize(grid, block)
      use ISO_C_BINDING
      integer (C_INT) :: meta_validate_worksize
      integer, dimension(3) :: grid, block
      meta_validate_worksize = meta_validate_worksize_c( &
       int(grid(1),kind=C_SIZE_T), int(grid(2),kind=C_SIZE_T), &
       int(grid(3),kind=C_SIZE_T), int(block(1),kind=C_SIZE_T), &
       int(block(2),kind=C_SIZE_T), int(block(3),kind=C_SIZE_T))
      end function meta_validate_worksize
      
      function meta_dotprod(grid, block, data1, data2, sz, start_idx&
      &, end_idx, reduction_var, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_dotprod
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
           meta_dotprod = meta_dotprod_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data1), C_LOC(data2), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), &
            int(4,kind=C_INT), int(async,kind=C_INT))
        end select
       end select
      type is(real(kind=C_FLOAT))
       select type (data1)
       type is (real(kind=C_FLOAT))
        select type (data2)
        type is (real(kind=C_FLOAT))
           meta_dotprod = meta_dotprod_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data1), C_LOC(data2), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), &
            int(1,kind=C_INT), int(async,kind=C_INT))
        end select
       end select
      type is(real(kind=C_DOUBLE))
       select type (data1)
       type is (real(kind=C_DOUBLE))
        select type (data2)
        type is (real(kind=C_DOUBLE))
           meta_dotprod = meta_dotprod_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data1), C_LOC(data2), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), &
            int(0,kind=C_INT), int(async,kind=C_INT))
        end select
       end select
      class default
            !do seomthing
      end select
      end function meta_dotprod
      
      function meta_reduce(grid, block, data, sz, start_idx&
      &, end_idx, reduction_var, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_reduce
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
           meta_reduce = meta_reduce_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), &
            int(4,kind=C_INT), int(async,kind=C_INT))
       end select
      type is(real(kind=C_FLOAT))
       select type (data)
       type is (real(kind=C_FLOAT))
           meta_reduce = meta_reduce_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), &
            int(1,kind=C_INT), int(async,kind=C_INT))
       end select
      type is(real(kind=C_DOUBLE))
       select type (data)
       type is (real(kind=C_DOUBLE))
           meta_reduce = meta_reduce_c(int(grid(1),kind=C_SIZE_T), &
            int(grid(2),kind=C_SIZE_T), int(grid(3),kind=C_SIZE_T), &
            int(block(1),kind=C_SIZE_T), int(block(2),kind=C_SIZE_T), &
            int(block(3),kind=C_SIZE_T), C_LOC(data), &
            int(sz(1),kind=C_SIZE_T), int(sz(2),kind=C_SIZE_T), &
            int(sz(3),kind=C_SIZE_T), int(start_idx(1),kind=C_SIZE_T),&
            int(start_idx(2),kind=C_SIZE_T), int(start_idx(3),kind=C_SIZE_T), &
            int(end_idx(1),kind=C_SIZE_T),int(end_idx(2),kind=C_SIZE_T)&
            , int(end_idx(3),kind=C_SIZE_T), C_LOC(reduction_var), &
            int(0,kind=C_INT), int(async,kind=C_INT))
       end select
      class default
            !do seomthing
      end select
      end function meta_reduce


      function meta_copy_h2d(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_h2d
      class(*), TARGET :: dst
      type(*), dimension(*), TARGET :: src
      integer :: size
      integer :: async
      select type (dst)
       type is (integer(kind=C_INT))
            meta_copy_h2d = meta_copy_h2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(in),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_FLOAT))
            meta_copy_h2d = meta_copy_h2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(fl),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_DOUBLE))
            meta_copy_h2d = meta_copy_h2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(db),kind=C_SIZE_T),int(async,kind=C_INT))
       class default
            ! do something
      end select
      end function meta_copy_h2d

      function meta_copy_d2h(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_d2h
      class(*), TARGET :: src 
      type(*), dimension(*), TARGET :: dst
      integer :: size
      integer :: async
      select type (src)
       type is (integer(kind=C_INT))
            meta_copy_d2h = meta_copy_d2h_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(in),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_FLOAT))
            meta_copy_d2h = meta_copy_d2h_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(fl),kind=C_SIZE_T),int(async,kind=C_INT))
       type is (real(kind=C_DOUBLE))
            meta_copy_d2h = meta_copy_d2h_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(db),kind=C_SIZE_T),int(async,kind=C_INT))
       class default
            ! do something
      end select
      end function meta_copy_d2h

      function meta_copy_d2d(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_d2d
      class(*), TARGET :: dst, src
      integer :: size
      integer :: async
      select type (dst)
       type is (integer(kind=C_INT))
       select type (src)
        type is (integer(kind=C_INT))
            meta_copy_d2d = meta_copy_d2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(in),kind=C_SIZE_T),int(async,kind=C_INT))
        end select
       type is (real(kind=C_FLOAT))
       select type (src)
        type is (real(kind=C_FLOAT))
            meta_copy_d2d = meta_copy_d2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(fl),kind=C_SIZE_T),int(async,kind=C_INT))
        end select
       type is (real(kind=C_DOUBLE))
       select type (src)
        type is (real(kind=C_DOUBLE))
            meta_copy_d2d = meta_copy_d2d_c(C_LOC(dst), C_LOC(src), &
             int(size*C_SIZEOF(db),kind=C_SIZE_T),int(async,kind=C_INT))
        end select
       class default
            ! do something
      end select
      end function meta_copy_d2d

      end module metamorph_c
