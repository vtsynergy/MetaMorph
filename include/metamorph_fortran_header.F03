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
