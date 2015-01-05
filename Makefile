MPICH_USE_SHLIB=yes

all: default notimers nocuda nocuda_notimers gpuDirect

default: red_test red_test_fortran

gpuDirect: xch_test_direct 

notimers: red_test_notimers red_test_fortran_notimers

nocuda: red_test_nocuda red_test_fortran_nocuda

nocuda_notimers: red_test_nocuda_notimers red_test_fortran_nocuda_notimers

debug: default_DEBUG notimers_DEBUG nocuda_DEBUG nocuda_notimers_DEBUG

default_DEBUG: red_test_DEBUG red_test_fortran_DEBUG xch_test_DEBUG

notimers_DEBUG: red_test_notimers_DEBUG red_test_fortran_notimers_DEBUG

nocuda_DEBUG: red_test_nocuda_DEBUG red_test_fortran_nocuda_DEBUG

nocuda_notimers_DEBUG: red_test_nocuda_notimers_DEBUG red_test_fortran_nocuda_notimers_DEBUG

xch_test: libafosr_cfd.so
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd -o xch_test -g /usr/lib/libnuma.so.1

xch_test_direct: libafosr_cfd_direct.so
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI_GPU_DIRECT -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_direct -o xch_test_direct -g /usr/lib/libnuma.so.1

red_test_fortran_nocuda_notimers: libafosr_cfd_nocuda_notimers.so
	gfortran -o red_test_fortran_nocuda_notimers red_bootstrapper.F03 -L ./ -lafosr_cfd_nocuda_notimers

red_test_fortran_nocuda: libafosr_cfd_nocuda.so
	gfortran -o red_test_fortran_nocuda red_bootstrapper.F03 -L ./ -lafosr_cfd_nocuda -D WITH_TIMERS

red_test_fortran_notimers: libafosr_cfd_notimers_fortran.so
	gfortran -o red_test_fortran_notimers red_bootstrapper.F03 -L ./ -lafosr_cfd_notimers_fortran

red_test_fortran: libafosr_cfd_fortran.so
	gfortran -o red_test_fortran red_bootstrapper.F03 -L ./ -lafosr_cfd_fortran -D WITH_TIMERS

red_test_nocuda_notimers: libafosr_cfd_nocuda_notimers.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_nocuda_notimers -lafosr_cfd_opencl_core -o red_test_nocuda_notimers

red_test_nocuda: libafosr_cfd_nocuda.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_nocuda -lafosr_cfd_opencl_core -o red_test_nocuda

red_test_notimers: libafosr_cfd_notimers.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_notimers -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -o red_test_notimers

red_test: libafosr_cfd.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd  -o red_test

libafosr_cfd_nocuda_notimers.so: libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lOpenCL -o libafosr_cfd_nocuda_notimers.so

libafosr_cfd_nocuda.so: libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c afosr_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lOpenCL -o libafosr_cfd_nocuda.so

libafosr_cfd_notimers.so: libafosr_cfd_cuda_core.so libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd_notimers.so

libafosr_cfd_notimers_fortran.so: libafosr_cfd_cuda_core.so libafosr_cfd_opencl_core.so afosr_cfd.c afosr_cfd_fortran_compat.c
	gcc afosr_cfd.c afosr_cfd_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd_notimers_fortran.so
#	OMPI_CC=gcc-4.8 /usr/bin/mpicc.openmpi afosr_cfd.c afosr_cfd_mpi.c afosr_cfd_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd_notimers.so /usr/lib/libnuma.so.1

#OMPI_CC=gcc-4.8 /usr/bin/mpicc
libafosr_cfd_direct.so: libafosr_cfd_cuda_core.so libafosr_cfd_opencl_core.so afosr_cfd.c
	mpicc -cc=gcc-4.8 afosr_cfd.c afosr_cfd_timers.c afosr_cfd_fortran_compat.c afosr_cfd_mpi.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -D WITH_MPI_GPU_DIRECT -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd_direct.so /usr/lib/libnuma.so.1

libafosr_cfd_fortran.so: libafosr_cfd_cuda_core.so libafosr_cfd_opencl_core.so afosr_cfd.c afosr_cfd_fortran_compat.c
	gcc afosr_cfd.c afosr_cfd_timers.c afosr_cfd_fortran_compat.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd_fortran.so



libafosr_cfd.so: libafosr_cfd_cuda_core.so libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd.so
	#mpicc -cc=gcc-4.8 afosr_cfd.c afosr_cfd_timers.c afosr_cfd_fortran_compat.c afosr_cfd_mpi.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd.so /usr/lib/libnuma.so.1

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libafosr_cfd_cuda_core.so:
	nvcc afosr_cfd_cuda_core.cu -Xcompiler -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lcudart  -o libafosr_cfd_cuda_core.so -D __CUDACC__ -arch sm_20

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libafosr_cfd_opencl_core.so:
	g++ afosr_cfd_opencl_core.cpp -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lOpenCL -o libafosr_cfd_opencl_core.so -D __OPENCLCC__

#Debug variants
red_test_fortran_nocuda_notimers_DEBUG: libafosr_cfd_nocuda_notimers_DEBUG.so
	gfortran -o red_test_fortran_nocuda_notimers_DEBUG red_bootstrapper.F03 -L ./ -lafosr_cfd_nocuda_notimers_DEBUG -g

red_test_fortran_nocuda_DEBUG: libafosr_cfd_nocuda.so
	gfortran -o red_test_fortran_nocuda_DEBUG red_bootstrapper.F03 -L ./ -lafosr_cfd_nocuda_DEBUG -D WITH_TIMERS -g

red_test_fortran_notimers_DEBUG: libafosr_cfd_notimers.so
	gfortran -o red_test_fortran_notimers_DEBUG red_bootstrapper.F03 -L ./ -lafosr_cfd_notimers_DEBUG -g

red_test_fortran_DEBUG: libafosr_cfd.so
	gfortran -o red_test_fortran_DEBUG red_bootstrapper.F03 -L ./ -lafosr_cfd_DEBUG -D WITH_TIMERS -g

red_test_nocuda_notimers_DEBUG: libafosr_cfd_nocuda_notimers_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_nocuda_notimers_DEBUG -lafosr_cfd_opencl_core_DEBUG -o red_test_nocuda_notimers_DEBUG -g -O0

red_test_nocuda_DEBUG: libafosr_cfd_nocuda_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_nocuda_DEBUG -lafosr_cfd_opencl_core_DEBUG -o red_test_nocuda_DEBUG -g -O0

red_test_notimers_DEBUG: libafosr_cfd_notimers_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_notimers_DEBUG -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -o red_test_notimers_DEBUG -g -O0

red_test_DEBUG: libafosr_cfd_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_DEBUG -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -o red_test_DEBUG -g -O0
xch_test_DEBUG:
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_DEBUG -o xch_test_DEBUG -g

libafosr_cfd_nocuda_notimers_DEBUG.so: libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core_DEBUG -lOpenCL -o libafosr_cfd_nocuda_notimers_DEBUG.so -g -O0

libafosr_cfd_nocuda_DEBUG.so: libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c afosr_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core_DEBUG -lOpenCL -o libafosr_cfd_nocuda_DEBUG.so -g -O0

libafosr_cfd_notimers_DEBUG.so: libafosr_cfd_cuda_core_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -lOpenCL -lcudart -o libafosr_cfd_notimers_DEBUG.so -g -O0

libafosr_cfd_DEBUG.so: libafosr_cfd_cuda_core_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c afosr_cfd_fortran_compat.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -lOpenCL -lcudart -o libafosr_cfd_DEBUG.so -g -O0

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libafosr_cfd_cuda_core_DEBUG.so:
	nvcc afosr_cfd_cuda_core.cu -Xcompiler -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lcudart  -o libafosr_cfd_cuda_core_DEBUG.so -D __CUDACC__ -arch sm_20 -g -O0

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libafosr_cfd_opencl_core_DEBUG.so:
	g++ afosr_cfd_opencl_core.cpp -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lOpenCL -o libafosr_cfd_opencl_core_DEBUG.so -D __OPENCLCC__ -g -O0
	

clean:
	rm libafosr*.so *.mod xch_test red_test red_test_nocuda red_test_notimers red_test_nocuda_notimers red_test_fortran red_test_fortran_nocuda red_test_fortran_notimers red_test_fortran_nocuda_notimers red_test_DEBUG red_test_nocuda_DEBUG red_test_notimers_DEBUG red_test_nocuda_notimers_DEBUG red_test_fortran_DEBUG red_test_fortran_nocuda_DEBUG red_test_fortran_notimers_DEBUG red_test_fortran_nocuda_notimers_DEBUG
