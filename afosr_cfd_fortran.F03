      module afosr_cfd_c
      interface accel_alloc
      function accel_alloc(ptr, size)
      use ISO_C_BINDING
      integer (C_INT) :: accel_alloc
      type (C_PTR), VALUE :: ptr
      integer (C_SIZE_T) :: size
      end function accel_alloc
      end interface

      interface accel_free
      function accel_free(ptr)
      use ISO_C_BINDING
      integer (C_INT) :: accel_free
      type (C_PTR), VALUE :: ptr
      end function accel_free
      end interface

      interface choose_accel
      function choose_accel(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: choose_accel
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function choose_accel
      end interface

      interface get_accel
      function get_accel(accel, mode)
      use ISO_C_BINDING
      integer (C_INT) :: get_accel
      integer (C_INT) :: accel
      integer (C_INT) :: mode
      end function get_accel
      end interface

      interface accel_validate_worksize
      function accel_validate_worksize(grid_x, grid_y, grid_z, block_x,&
      & block_y, block_z)
      use ISO_C_BINDING
      integer (C_INT) :: accel_validate_worksize
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      end function accel_validate_worksize
      end interface

      interface accel_dotprod
      function accel_dotprod(grid_x, grid_y, grid_z, block_x, block_y,&
      & block_z, data1, data2, size_x, size_y, size_z, start_x, start_y&
      &, start_z, end_x, end_y, end_z, reduction_var, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_dotprod
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      type (C_PTR), VALUE :: data1, data2
      integer (C_SIZE_T) :: size_x, size_y, size_z
      integer (C_SIZE_T) :: start_x, start_y, start_z
      integer (C_SIZE_T) :: end_x, end_y, end_z
      type (C_PTR), VALUE :: reduction_var
      integer (C_INT) :: async
      end function accel_dotprod
      end interface

      interface accel_reduce
      function accel_reduce(grid_x, grid_y, grid_z, block_x, block_y,&
      & block_z, data, size_x, size_y, size_z, start_x, start_y&
      &, start_z, end_x, end_y, end_z, reduction_var, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_dotprod
      integer (C_SIZE_T) :: grid_x, grid_y, grid_z
      integer (C_SIZE_T) :: block_x, block_y, block_z
      type (C_PTR), VALUE :: data
      integer (C_SIZE_T) :: size_x, size_y, size_z
      integer (C_SIZE_T) :: start_x, start_y, start_z
      integer (C_SIZE_T) :: end_x, end_y, end_z
      type (C_PTR), VALUE :: reduction_var
      integer (C_INT) :: async
      end function accel_reduce
      end interface

      interface accel_copy_h2d
      function accel_copy_h2d(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_h2d
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function accel_copy_h2d
      end interface

      interface accel_copy_d2h
      function accel_copy_d2h(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_d2h
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function accel_copy_d2h
      end interface

      interface accel_copy_d2d
      function accel_copy_d2d(dst, src, size, async)
      use ISO_C_BINDING
      integer (C_INT) :: accel_copy_d2d
      type (C_PTR), VALUE :: dst, src
      integer (C_SIZE_T) :: size
      integer (C_INT) :: async
      end function accel_copy_d2d
      end interface

#ifdef WITH_TIMERS

      interface accel_timers_init
      function accel_timers_init()
      use ISO_C_BINDING
      integer (C_INT) :: accel_timers_init
      end function accel_timers_init
      end interface

      interface accel_timers_flush
      function accel_timers_flush()
      use ISO_C_BINDING
      integer (C_INT) :: accel_timers_flush
      end function accel_timers_flush
      end interface

      interface accel_timers_finish
      function accel_timers_finish()
      use ISO_C_BINDING
      integer (C_INT) :: accel_timers_finish
      end function accel_timers_finish
      end interface
      
#endif

      end module afosr_cfd_c
