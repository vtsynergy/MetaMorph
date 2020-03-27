!> @file
!> For detailed function documentation, please consult the C headers for the wrapped function
      module metamorph_c
      use ISO_C_BINDING
!> Ensure the double-precision implementation is a C double
      real (C_DOUBLE) :: db
!> Ensure the single-precision implementation is a C float
      real (C_FLOAT) :: fl
!> Ensure the integer implementation is a C 4-byte integer
      integer (C_INT) :: in

!> Fortran interface for creating a MetaMorph buffer
      interface meta_alloc_c
!> The function prototype of meta_alloc_c, which matches the C implementation
!> @param ptr An address to return the c_ptr (a MetaMorph void * handle) in
!> @param size The number of bytes to allocate
!> @return The error status of the underlying meta_alloc call
      function meta_alloc_c(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: meta_alloc_c
      type (C_PTR) :: ptr
      integer (C_SIZE_T) :: size
      end function meta_alloc_c
      end interface

!> Fortran interface for releasing a MetaMorph buffer
      interface meta_free_c
!> The function prototype of meta_free_c which matches the C implementation
!> @param ptr The device buffer to release, a MetaMorph handle returned from meta_alloc_c
!> @return The error status of the underlying meta_free call
      function meta_free_c(ptr)
      use ISO_C_BINDING
      integer (C_INT) :: meta_free_c
      type (C_PTR), VALUE :: ptr
      end function meta_free_c
      end interface

!> Fortran interface for setting the current accelerator and backend
      interface meta_set_acc
!> The function prototype of meta_set_acc_c which matches the C implementation
!> @param accel The desired accelerator's ID on the desired backend
!> @param mode The desired backend mode to switch to
!> @return The error status of the underlying meta_set_acc call
      function meta_set_acc_c(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: meta_set_acc
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function meta_set_acc_c
      end interface

!> Fortran interface for getting the current accelerator and backend
      interface meta_get_acc
!> The function prototype of meta_get_acc_c which matches the C implementation
!> @param accel Address in which to return the currently-active device's ID within the current backend
!> @param mode Address in which to return the currently-active backend mode
!> @return The error status of the underlying meta_get_acc call
      function meta_get_acc_c(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: meta_get_acc
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function meta_get_acc_c
      end interface

!> Fortran interface for worksize validation
      interface meta_validate_worksize_c
!> The function prototype of meta_validate_worksize_c which matches the C implementation
!> @param grid_x Number of desired thread blocks in the X dimension
!> @param grid_y Number of desired thread blocks in the Y dimension
!> @param grid_z Number of desired thread blocks in the Z dimension
!> @param block_x Number of desired threads within each block in the X dimension
!> @param block_y Number of desired threads within each block in the Y dimension
!> @param block_z Number of desired threads within each block in the Z dimension
!> @return The error status of the underlying meta_validate_worksize call
      function meta_validate_worksize_c(grid_x, grid_y, grid_z, &
      & block_x, block_y, block_z)
      use ISO_C_BINDING
      integer (C_INT) :: meta_validate_worksize
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      end function meta_validate_worksize_c
      end interface

!> Fortran interface for a dot product
      interface meta_dotprod_c
!> The function prototype of meta_dotprod_c, which matches the C implementation
!> @param grid_x Number of desired thread blocks in the X dimension
!> @param grid_y Number of desired thread blocks in the Y dimension
!> @param grid_z Number of desired thread blocks in the Z dimension
!> @param block_x Number of desired threads within each block in the X dimension
!> @param block_y Number of desired threads within each block in the Y dimension
!> @param block_z Number of desired threads within each block in the Z dimension
!> @param data1 The left matrix in the dot product operator, a Metamorph-allocated buffer on the current device (a MetaMorph void * handle)
!> @param data2 The right matrix in the dot product operator, a Metamorph-allocated buffer on the current device (a MetaMorph void * handle)
!> @param size_x Number of elements in data1 and data2 in the X dimension
!> @param size_y Number of elements in data1 and data2 in the Y dimension
!> @param size_z Number of elements in data1 and data2 in the Z dimension
!> @param start_x Index in data1 and data2 of the first element in the X dimension
!> @param start_y Index in data1 and data2 of the first element in the Y dimension
!> @param start_z Index in data1 and data2 of the first element in the Z dimension
!> @param end_x Index in data1 and data2 of the last element in the X dimension
!> @param end_y Index in data1 and data2 of the last element in the Y dimension
!> @param end_z Index in data1 and data2 of the last element in the Z dimension
!> @param reduction_var The final scalar dot product value, a Metamorph-allocated buffer on the current device (a MetaMorph void * handle)
!> @param a_type The MetaMorph data type to interpret the data arrays as
!> @param async Whether the kernel should be run asynchronously or blocking
!> @return The error status of the underlying meta_dotProd call
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

!> Fortran interface for a reduction sum
      interface meta_reduce_c
!> The function prototype of meta_dotprod_c, which matches the C implementation
!> @param grid_x Number of desired thread blocks in the X dimension
!> @param grid_y Number of desired thread blocks in the Y dimension
!> @param grid_z Number of desired thread blocks in the Z dimension
!> @param block_x Number of desired threads within each block in the X dimension
!> @param block_y Number of desired threads within each block in the Y dimension
!> @param block_z Number of desired threads within each block in the Z dimension
!> @param data The buffer to perform sum-reduction on, a Metamorph-allocated buffer on the current device (a MetaMorph void * handle)
!> @param size_x Number of elements in data in the X dimension
!> @param size_y Number of elements in data in the Y dimension
!> @param size_z Number of elements in data in the Z dimension
!> @param start_x Index in data of the first element in the X dimension
!> @param start_y Index in data of the first element in the Y dimension
!> @param start_z Index in data of the first element in the Z dimension
!> @param end_x Index in data of the last element in the X dimension
!> @param end_y Index in data of the last element in the Y dimension
!> @param end_z Index in data of the last element in the Z dimension
!> @param reduction_var The final scalar reduction sum value, a Metamorph-allocated buffer on the current device (a MetaMorph void * handle)
!> @param a_type The MetaMorph data type to interpret the data arrays as
!> @param async Whether the kernel should be run asynchronously or blocking
!> @return The error status of the underlying meta_reduce call
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

!> Fortran interface for writing to a MetaMorph buffer on the current device
      interface meta_copy_h2d_c
!> The function prototype of meta_copy_h2d_c which matches the C implementation
!> @param dst The device destination buffer, a MetaMorph handle returned from meta_alloc_c
!> @param src The dhost source buffer
!> @param size The number of bytes to write to the device
!> @param async whether to perform the write asynchronously or blocking
!> @return The error status of the underlying meta_copy_h2d call
      function meta_copy_h2d_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_h2d_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function meta_copy_h2d_c
      end interface

!> Fortran interface for reading a MetaMorph buffer from the current device
      interface meta_copy_d2h_c
!> The function prototype of meta_copy_d2h_c which matches the C implementation
!> @param dst The host destination buffer
!> @param src The device source buffer, a MetaMorph handle returned from meta_alloc_c
!> @param size The number of bytes to read fromn the device
!> @param async whether to perform the read asynchronously or blocking
!> @return The error status of the underlying meta_copy_d2h call
      function meta_copy_d2h_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_d2h_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function meta_copy_d2h_c
      end interface

!> Fortran interface for copying between two MetaMorph buffers on the current device
      interface meta_copy_d2d_c
!> The function prototype of meta_copy_d2d_c which matches the C implementation
!> @param dst The device destination buffer, a MetaMorph handle returned from meta_alloc_c
!> @param src The device source buffer, a MetaMorph handle returned from meta_alloc_c
!> @param size The number of bytes to copy on the device
!> @param async whether to perform the copy asynchronously or blocking
!> @return The error status of the underlying meta_copy_d2d call
      function meta_copy_d2d_c(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: meta_copy_d2d_c
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function meta_copy_d2d_c
      end interface

!> Fortran interface around the three most common types to allocate
      interface meta_alloc
        module procedure meta_alloc_db, meta_alloc_fl, meta_alloc_in
      end interface meta_alloc

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
