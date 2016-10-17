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

xch_test: libmetamorph.so
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph -o xch_test -g
#/usr/lib/libnuma.so.1

xch_test_direct: libmetamorph_direct.so
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -fopenmp -pg -g -D__DEBUG__ -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI_GPU_DIRECT -L ./ -L /usr/local/cuda/lib64 -lmetamorph_direct -o xch_test_direct -g
#/usr/lib/libnuma.so.1
	#mpicc -cc=gcc-4.8 exchange_bootstrapper.c -g -D__DEBUG__ -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI_GPU_DIRECT -L ./ -L /usr/local/cuda/lib64 -lmetamorph_direct -o xch_test_direct -g /usr/lib/libnuma.so.1

red_test_fortran_nocuda_notimers: libmetamorph_nocuda_notimers.so
	gfortran -o red_test_fortran_nocuda_notimers red_bootstrapper.F03 -fopenmp -L ./ -lmetamorph_nocuda_notimers -lmm_openmp_backend

red_test_fortran_nocuda: libmetamorph_nocuda.so
	gfortran -o red_test_fortran_nocuda red_bootstrapper.F03 -fopenmp -L ./ -lmetamorph_nocuda -lmm_openmp_backend -DWITH_OPENMP -D WITH_TIMERS

red_test_fortran_notimers: libmetamorph_notimers_fortran.so
	gfortran -o red_test_fortran_notimers red_bootstrapper.F03 -fopenmp -L ./ -lmetamorph_notimers_fortran

red_test_fortran: libmetamorph_fortran.so
	gfortran -o red_test_fortran red_bootstrapper.F03 -L ./ -lmetamorph_fortran -D WITH_TIMERS -fopenmp -D WITH_OPENMP -DWITH_CUDA -DWITH_OPENCL

red_test_nocuda_notimers: libmetamorph_nocuda_notimers.so
	gcc red_bootstrapper.c -fopenmp -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_nocuda_notimers -lmm_openmp_backend -lmm_opencl_backend -o red_test_nocuda_notimers

red_test_nocuda: libmetamorph_nocuda.so
	gcc red_bootstrapper.c -fopenmp -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_nocuda -lmm_openmp_backend -lmm_opencl_backend -o red_test_nocuda

red_test_notimers: libmetamorph_notimers.so
	gcc red_bootstrapper.c -I ./ -fopenmp -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -L ./ -L /usr/local/cuda/lib64 -lmetamorph_notimers -lmm_openmp_backend -lmm_opencl_backend -lmm_cuda_backend -o red_test_notimers

red_test: libmetamorph.so
	gcc red_bootstrapper.c -fopenmp -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -L ./ -L /usr/local/cuda/lib64 -lmetamorph  -o red_test

libmetamorph_nocuda_notimers.so: libmm_opencl_backend.so metamorph.c
	gcc metamorph.c metamorph_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lOpenCL -o libmetamorph_nocuda_notimers.so

libmetamorph_nocuda.so: libmm_opencl_backend.so metamorph.c
	gcc metamorph.c metamorph_timers.c metamorph_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lOpenCL -o libmetamorph_nocuda.so

libmetamorph_notimers.so: libmm_cuda_backend.so libmm_opencl_backend.so metamorph.c
	mpicc -cc=gcc-4.8 metamorph.c metamorph_mpi.c -g -DWITH_MPI -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph_notimers.so -DWITH_MPI_POOL_TIMING -DWITH_MPI_POOL
#	gcc metamorph.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph_notimers.so

libmetamorph_notimers_fortran.so: libmm_cuda_backend.so libmm_opencl_backend.so metamorph.c metamorph_fortran_compat.c
	gcc metamorph.c metamorph_fortran_compat.c -fopenmp -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_openmp_backend -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph_notimers_fortran.so
#	OMPI_CC=gcc-4.8 /usr/bin/mpicc.openmpi metamorph.c metamorph_mpi.c metamorph_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph_notimers.so /usr/lib/libnuma.so.1

#OMPI_CC=gcc-4.8 /usr/bin/mpicc
libmetamorph_direct.so: libmm_cuda_backend.so libmm_opencl_backend.so metamorph.c
	which mpicc
	mpicc -cc=gcc-4.8 metamorph.c metamorph_timers.c metamorph_fortran_compat.c metamorph_mpi.c -fopenmp -g -fPIC -shared -DNO_FIXME -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -D WITH_MPI_GPU_DIRECT -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_openmp_backend -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph_direct.so -DWITH_MPI_POOL_TIMING -DWITH_MPI_POOL
# /usr/lib/libnuma.so.1
	#mpicc -cc=gcc-4.8 metamorph.c metamorph_timers.c metamorph_fortran_compat.c metamorph_mpi.c -g -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -D WITH_MPI -D WITH_MPI_GPU_DIRECT -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph_direct.so /usr/lib/libnuma.so.1

libmetamorph_fortran.so: libmm_cuda_backend.so libmm_opencl_backend.so metamorph.c metamorph_fortran_compat.c
	gcc metamorph.c metamorph_timers.c metamorph_fortran_compat.c -fopenmp -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_openmp_backend -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph_fortran.so



libmetamorph.so: libmm_cuda_backend.so libmm_opencl_backend.so libmm_openmp_backend.so metamorph.c
	gcc metamorph.c metamorph_timers.c -fopenmp -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lmm_cuda_backend -lmm_openmp_backend -lOpenCL -lcudart -o libmetamorph.so
	#mpicc -cc=gcc-4.8 metamorph.c metamorph_timers.c metamorph_fortran_compat.c metamorph_mpi.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend -lmm_cuda_backend -lOpenCL -lcudart -o libmetamorph.so /usr/lib/libnuma.so.1

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libmm_cuda_backend.so:
	nvcc mm_cuda_backend.cu -O3 -Xcompiler -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lcudart  -o libmm_cuda_backend.so -D __CUDACC__ -arch sm_20

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libmm_opencl_backend.so:
	g++ mm_opencl_backend.cpp -O3 -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lOpenCL -o libmm_opencl_backend.so -D __OPENCLCC__


libmm_openmp_backend.so:
	gcc mm_openmp_backend.c -O3 -fopenmp -fPIC -shared -I ./ -o libmm_openmp_backend.so
#Debug variants
red_test_fortran_nocuda_notimers_DEBUG: libmetamorph_nocuda_notimers_DEBUG.so
	gfortran -o red_test_fortran_nocuda_notimers_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_nocuda_notimers_DEBUG -g

red_test_fortran_nocuda_DEBUG: libmetamorph_nocuda.so
	gfortran -o red_test_fortran_nocuda_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_nocuda_DEBUG -D WITH_TIMERS -g

red_test_fortran_notimers_DEBUG: libmetamorph_notimers.so
	gfortran -o red_test_fortran_notimers_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_notimers_DEBUG -g

red_test_fortran_DEBUG: libmetamorph.so
	gfortran -o red_test_fortran_DEBUG red_bootstrapper.F03 -L ./ -lmetamorph_DEBUG -D WITH_TIMERS -g

red_test_nocuda_notimers_DEBUG: libmetamorph_nocuda_notimers_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_nocuda_notimers_DEBUG -lmm_opencl_backend_DEBUG -o red_test_nocuda_notimers_DEBUG -g -O0

red_test_nocuda_DEBUG: libmetamorph_nocuda_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_nocuda_DEBUG -lmm_opencl_backend_DEBUG -o red_test_nocuda_DEBUG -g -O0

red_test_notimers_DEBUG: libmetamorph_notimers_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_notimers_DEBUG -lmm_opencl_backend_DEBUG -lmm_cuda_backend_DEBUG -o red_test_notimers_DEBUG -g -O0

red_test_DEBUG: libmetamorph_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -L ./ -L /usr/local/cuda/lib64 -lmetamorph_DEBUG -lmm_opencl_backend_DEBUG -lmm_cuda_backend_DEBUG -o red_test_DEBUG -g -O0
xch_test_DEBUG:
	mpicc -cc=gcc-4.8 exchange_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_CUDA -D WITH_MPI -D WITH_TIMERS -D WITH_FORTRAN -L ./ -L /usr/local/cuda/lib64 -lmetamorph_DEBUG -o xch_test_DEBUG -g

libmetamorph_nocuda_notimers_DEBUG.so: libmm_opencl_backend_DEBUG.so metamorph.c
	gcc metamorph.c metamorph_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend_DEBUG -lOpenCL -o libmetamorph_nocuda_notimers_DEBUG.so -g -O0

libmetamorph_nocuda_DEBUG.so: libmm_opencl_backend_DEBUG.so metamorph.c
	gcc metamorph.c metamorph_timers.c metamorph_fortran_compat.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend_DEBUG -lOpenCL -o libmetamorph_nocuda_DEBUG.so -g -O0

libmetamorph_notimers_DEBUG.so: libmm_cuda_backend_DEBUG.so libmm_opencl_backend_DEBUG.so metamorph.c
	gcc metamorph.c metamorph_fortran_compat.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_FORTRAN -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend_DEBUG -lmm_cuda_backend_DEBUG -lOpenCL -lcudart -o libmetamorph_notimers_DEBUG.so -g -O0

libmetamorph_DEBUG.so: libmm_cuda_backend_DEBUG.so libmm_opencl_backend_DEBUG.so metamorph.c
	gcc metamorph.c metamorph_timers.c metamorph_fortran_compat.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -D WITH_FORTRAN -D WITH_MPI -I ./ -L ./ -L /usr/local/cuda/lib64 -lmm_opencl_backend_DEBUG -lmm_cuda_backend_DEBUG -lOpenCL -lcudart -o libmetamorph_DEBUG.so -g -O0

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libmm_cuda_backend_DEBUG.so:
	nvcc mm_cuda_backend.cu -Xcompiler -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lcudart  -o libmm_cuda_backend_DEBUG.so -D __CUDACC__ -arch sm_20 -g -O0

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libmm_opencl_backend_DEBUG.so:
	g++ mm_opencl_backend.cpp -fPIC -shared -I ./ -L /usr/local/cuda/lib64 -lOpenCL -o libmm_opencl_backend_DEBUG.so -D __OPENCLCC__ -g -O0
	

clean:
	rm libmetamorph*.so *.mod xch_test red_test red_test_nocuda red_test_notimers red_test_nocuda_notimers red_test_fortran red_test_fortran_nocuda red_test_fortran_notimers red_test_fortran_nocuda_notimers red_test_DEBUG red_test_nocuda_DEBUG red_test_notimers_DEBUG red_test_nocuda_notimers_DEBUG red_test_fortran_DEBUG red_test_fortran_nocuda_DEBUG red_test_fortran_notimers_DEBUG red_test_fortran_nocuda_notimers_DEBUG
