include "metamorph_fortran_header.F03"

      contains
!> Fortran implemetation for allocating a MetaMorph buffer of double-precision values
      function meta_alloc_db(ptr, size)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_alloc call
      integer (C_INT) :: meta_alloc_db
      real(C_DOUBLE), POINTER, intent(inout) :: ptr
      integer :: size
      type(C_PTR) :: cptr
      meta_alloc_db = meta_alloc_c(cptr, &
       int(C_SIZEOF(db)*size, kind=C_SIZE_T))
      call C_F_POINTER(cptr, ptr)
      end function meta_alloc_db

!> Fortran implementation for allocating a MetaMorph buffer of single-precision values
      function meta_alloc_fl(ptr, size)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_alloc call
      integer (C_INT) :: meta_alloc_fl
      real(C_FLOAT), POINTER, intent(inout) :: ptr
      integer :: size
      type(C_PTR) :: cptr
      meta_alloc_fl = meta_alloc_c(cptr, &
       int(C_SIZEOF(fl)*size, kind=C_SIZE_T))
      call C_F_POINTER(cptr, ptr) 
      end function meta_alloc_fl

!> Fortran implementation for allocating a MetaMorph buffer of 4-byte integer values
      function meta_alloc_in(ptr, size)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_alloc call
      integer (C_INT) :: meta_alloc_in
      integer(C_INT), POINTER, intent(inout) :: ptr
      integer :: size
      type(C_PTR) :: cptr
      meta_alloc_in = meta_alloc_c(cptr, &
       int(C_SIZEOF(in)*size, kind=C_SIZE_T))
      call C_F_POINTER(cptr, ptr)
      end function meta_alloc_in

!> Fortran implementation for releasing a MetaMorph buffer
      function meta_free(ptr)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_free call
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

!> Fortran implementation around the worksize validator
      function meta_validate_worksize(grid, block)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_validate_worksize call
      integer (C_INT) :: meta_validate_worksize
      integer, dimension(3) :: grid, block
      meta_validate_worksize = meta_validate_worksize_c( &
       int(grid(1),kind=C_SIZE_T), int(grid(2),kind=C_SIZE_T), &
       int(grid(3),kind=C_SIZE_T), int(block(1),kind=C_SIZE_T), &
       int(block(2),kind=C_SIZE_T), int(block(3),kind=C_SIZE_T))
      end function meta_validate_worksize
      
!> Fortran implementation around the dot product kernel
      function meta_dotprod(grid, block, data1, data2, sz, start_idx&
      &, end_idx, reduction_var, async)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_dotprod call
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
      
!> Fortran implementation around the reduction sum kernel
      function meta_reduce(grid, block, data, sz, start_idx&
      &, end_idx, reduction_var, async)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_reduce call
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

!> Fortran implementation around the function to write to a device buffer
      function meta_copy_h2d(dst, src, size, async)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_copy_h2d call
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

!> Fortran implementation around the function to read from a device buffer
      function meta_copy_d2h(dst, src, size, async)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_copy_d2h call
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

!> Fortran implementation around the function to copy between two device buffers
      function meta_copy_d2d(dst, src, size, async)
      use ISO_C_BINDING
!> @return The error status of the underlying meta_copy_d2d call
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
