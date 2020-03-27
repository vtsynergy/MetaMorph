!> @file
!! Fortran wrappers around MetaMorph's Event-based profiling plugin
      module metamorph_profiling_c
      use ISO_C_BINDING

!> Explicitly initialize the profiling infrastructure
      interface meta_timers_init
!> Reference the Fortran entrypoint for the C function
      subroutine meta_timers_init_c()
      use ISO_C_BINDING
      end subroutine meta_timers_init_c
      end interface

!> Flush all oustanding timers but keep internal queues intact
      interface meta_timers_flush
!> Reference the Fortran entrypoint for the C function
      subroutine meta_timers_flush_c()
      use ISO_C_BINDING
      end subroutine meta_timers_flush_c
      end interface

!> Flush all outstanding profiling data and destruct
      interface meta_timers_finish
!> Reference the Fortran entrypoint for the C function
      subroutine meta_timers_finish_c()
      use ISO_C_BINDING
      end subroutine meta_timers_finish_c
      end interface

      end module metamorph_profiling_c
