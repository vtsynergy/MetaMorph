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

xch_test: libmetamorph_cfd.so
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd -o xch_test -g /usr/lib/libnuma.so.1

xch_test_direct: libmetamorph_cfd_direct.so
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -pg -g -D__DEBUG__ -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI_GPU_DIRECT -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_direct -o xch_test_direct -g /usr/lib/libnuma.so.1
	#mpicc -cc=gcc-4.8 exchange_bootstrapper.c -g -D__DEBUG__ -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI_GPU_DIRECT -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_direct -o xch_test_direct -g /usr/lib/libnuma.so.1

red_test_fortran_nocuda_notimers: libmetamorph_cfd_nocuda_notimers.so
	gfortran -o red_test_fortran_nocuda_notimers red_bootstrapper.F03 -L ./ -lmetamorph_cfd_nocuda_notimers

red_test_fortran_nocuda: libmetamorph_cfd_nocuda.so
	gfortran -o red_test_fortran_nocuda red_bootstrapper.F03 -L ./ -lmetamorph_cfd_nocuda -D WITH_TIMERS

red_test_fortran_notimers: libmetamorph_cfd_notimers_fortran.so
	gfortran -o red_test_fortran_notimers red_bootstrapper.F03 -L ./ -lmetamorph_cfd_notimers_fortran

red_test_fortran: libmetamorph_cfd_fortran.so
	gfortran -o red_test_fortran red_bootstrapper.F03 -L ./ -lmetamorph_cfd_fortran -D WITH_TIMERS

red_test_nocuda_notimers: libmetamorph_cfd_nocuda_notimers.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_nocuda_notimers -lmetamorph_cfd_opencl_core -o red_test_nocuda_notimers

red_test_nocuda: libmetamorph_cfd_nocuda.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_nocuda -lmetamorph_cfd_opencl_core -o red_test_nocuda

red_test_notimers: libmetamorph_cfd_notimers.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_notimers -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -o red_test_notimers

red_test: libmetamorph_cfd.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd  -o red_test

libmetamorph_cfd_nocuda_notimers.so: libmetamorph_cfd_opencl_core.so metamorph_cfd.c
	gcc metamorph_cfd.c metamorph_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lOpenCL -o libmetamorph_cfd_nocuda_notimers.so

libmetamorph_cfd_nocuda.so: libmetamorph_cfd_opencl_core.so metamorph_cfd.c
	gcc metamorph_cfd.c metamorph_cfd_timers.c metamorph_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lOpenCL -o libmetamorph_cfd_nocuda.so

libmetamorph_cfd_notimers.so: libmetamorph_cfd_cuda_core.so libmetamorph_cfd_opencl_core.so metamorph_cfd.c
	mpicc -cc=gcc-4.8 metamorph_cfd.c metamorph_cfd_mpi.c -g -DWITH_MPI -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd_notimers.so
#	gcc metamorph_cfd.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd_notimers.so

libmetamorph_cfd_notimers_fortran.so: libmetamorph_cfd_cuda_core.so libmetamorph_cfd_opencl_core.so metamorph_cfd.c metamorph_cfd_fortran_compat.c
	gcc metamorph_cfd.c metamorph_cfd_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd_notimers_fortran.so
#	OMPI_CC=gcc-4.8 /usr/bin/mpicc.openmpi metamorph_cfd.c metamorph_cfd_mpi.c metamorph_cfd_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd_notimers.so /usr/lib/libnuma.so.1

#OMPI_CC=gcc-4.8 /usr/bin/mpicc
libmetamorph_cfd_direct.so: libmetamorph_cfd_cuda_core.so libmetamorph_cfd_opencl_core.so metamorph_cfd.c
	which mpicc
	mpicc -cc=gcc-4.8 metamorph_cfd.c metamorph_cfd_timers.c metamorph_cfd_fortran_compat.c metamorph_cfd_mpi.c -g -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd_direct.so /usr/lib/libnuma.so.1
	#mpicc -cc=gcc-4.8 metamorph_cfd.c metamorph_cfd_timers.c metamorph_cfd_fortran_compat.c metamorph_cfd_mpi.c -g -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -D WITH_MPI_GPU_DIRECT -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd_direct.so /usr/lib/libnuma.so.1

libmetamorph_cfd_fortran.so: libmetamorph_cfd_cuda_core.so libmetamorph_cfd_opencl_core.so metamorph_cfd.c metamorph_cfd_fortran_compat.c
	gcc metamorph_cfd.c metamorph_cfd_timers.c metamorph_cfd_fortran_compat.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd_fortran.so



libmetamorph_cfd.so: libmetamorph_cfd_cuda_core.so libmetamorph_cfd_opencl_core.so metamorph_cfd.c
	gcc metamorph_cfd.c metamorph_cfd_timers.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd.so
	#mpicc -cc=gcc-4.8 metamorph_cfd.c metamorph_cfd_timers.c metamorph_cfd_fortran_compat.c metamorph_cfd_mpi.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core -lmetamorph_cfd_cuda_core -lOpenCL -lcudart -o libmetamorph_cfd.so /usr/lib/libnuma.so.1

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libmetamorph_cfd_cuda_core.so:
	nvcc metamorph_cfd_cuda_core.cu -g -Xcompiler -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lcudart  -o libmetamorph_cfd_cuda_core.so -D __CUDACC__ -arch sm_20

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libmetamorph_cfd_opencl_core.so:
	g++ metamorph_cfd_opencl_core.cpp -g -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lOpenCL -o libmetamorph_cfd_opencl_core.so -D __OPENCLCC__

#Debug variants
red_test_fortran_nocuda_notimers_DEBUG: libmetamorph_cfd_nocuda_notimers_DEBUG.so
	gfortran -o red_test_fortran_nocuda_notimers_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_cfd_nocuda_notimers_DEBUG -g

red_test_fortran_nocuda_DEBUG: libmetamorph_cfd_nocuda.so
	gfortran -o red_test_fortran_nocuda_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_cfd_nocuda_DEBUG -D WITH_TIMERS -g

red_test_fortran_notimers_DEBUG: libmetamorph_cfd_notimers.so
	gfortran -o red_test_fortran_notimers_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_cfd_notimers_DEBUG -g

red_test_fortran_DEBUG: libmetamorph_cfd.so
	gfortran -o red_test_fortran_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_cfd_DEBUG -D WITH_TIMERS -g

red_test_nocuda_notimers_DEBUG: libmetamorph_cfd_nocuda_notimers_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_nocuda_notimers_DEBUG -lmetamorph_cfd_opencl_core_DEBUG -o red_test_nocuda_notimers_DEBUG -g -O0

red_test_nocuda_DEBUG: libmetamorph_cfd_nocuda_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_nocuda_DEBUG -lmetamorph_cfd_opencl_core_DEBUG -o red_test_nocuda_DEBUG -g -O0

red_test_notimers_DEBUG: libmetamorph_cfd_notimers_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_notimers_DEBUG -lmetamorph_cfd_opencl_core_DEBUG -lmetamorph_cfd_cuda_core_DEBUG -o red_test_notimers_DEBUG -g -O0

red_test_DEBUG: libmetamorph_cfd_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_DEBUG -lmetamorph_cfd_opencl_core_DEBUG -lmetamorph_cfd_cuda_core_DEBUG -o red_test_DEBUG -g -O0
xch_test_DEBUG:
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_DEBUG -o xch_test_DEBUG -g

libmetamorph_cfd_nocuda_notimers_DEBUG.so: libmetamorph_cfd_opencl_core_DEBUG.so metamorph_cfd.c
	gcc metamorph_cfd.c metamorph_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core_DEBUG -lOpenCL -o libmetamorph_cfd_nocuda_notimers_DEBUG.so -g -O0

libmetamorph_cfd_nocuda_DEBUG.so: libmetamorph_cfd_opencl_core_DEBUG.so metamorph_cfd.c
	gcc metamorph_cfd.c metamorph_cfd_timers.c metamorph_cfd_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core_DEBUG -lOpenCL -o libmetamorph_cfd_nocuda_DEBUG.so -g -O0

libmetamorph_cfd_notimers_DEBUG.so: libmetamorph_cfd_cuda_core_DEBUG.so libmetamorph_cfd_opencl_core_DEBUG.so metamorph_cfd.c
	gcc metamorph_cfd.c metamorph_cfd_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core_DEBUG -lmetamorph_cfd_cuda_core_DEBUG -lOpenCL -lcudart -o libmetamorph_cfd_notimers_DEBUG.so -g -O0

libmetamorph_cfd_DEBUG.so: libmetamorph_cfd_cuda_core_DEBUG.so libmetamorph_cfd_opencl_core_DEBUG.so metamorph_cfd.c
	gcc metamorph_cfd.c metamorph_cfd_timers.c metamorph_cfd_fortran_compat.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lmetamorph_cfd_opencl_core_DEBUG -lmetamorph_cfd_cuda_core_DEBUG -lOpenCL -lcudart -o libmetamorph_cfd_DEBUG.so -g -O0

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libmetamorph_cfd_cuda_core_DEBUG.so:
	nvcc metamorph_cfd_cuda_core.cu -Xcompiler -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lcudart  -o libmetamorph_cfd_cuda_core_DEBUG.so -D __CUDACC__ -arch sm_20 -g -O0

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libmetamorph_cfd_opencl_core_DEBUG.so:
	g++ metamorph_cfd_opencl_core.cpp -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lOpenCL -o libmetamorph_cfd_opencl_core_DEBUG.so -D __OPENCLCC__ -g -O0
	

clean:
	rm libmetamorph*.so *.mod xch_test red_test red_test_nocuda red_test_notimers red_test_nocuda_notimers red_test_fortran red_test_fortran_nocuda red_test_fortran_notimers red_test_fortran_nocuda_notimers red_test_DEBUG red_test_nocuda_DEBUG red_test_notimers_DEBUG red_test_nocuda_notimers_DEBUG red_test_fortran_DEBUG red_test_fortran_nocuda_DEBUG red_test_fortran_notimers_DEBUG red_test_fortran_nocuda_notimers_DEBUG
