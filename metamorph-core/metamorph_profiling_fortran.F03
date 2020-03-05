      module metamorph_profiling_c
      use ISO_C_BINDING

      interface meta_timers_init
      function meta_timers_init_c()
      use ISO_C_BINDING
      integer (C_INT) :: meta_timers_init_c
      end function meta_timers_init_c
      end interface

      interface meta_timers_flush
      function meta_timers_flush_c()
      use ISO_C_BINDING
      integer (C_INT) :: meta_timers_flush_c
      end function meta_timers_flush_c
      end interface

      interface meta_timers_finish
      function meta_timers_finish_c()
      use ISO_C_BINDING
      integer (C_INT) :: meta_timers_finish_c
      end function meta_timers_finish_c
      end interface

      end module metamorph_profiling_c
