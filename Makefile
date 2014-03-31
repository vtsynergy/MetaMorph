all: red_test notimers nocuda nocuda_notimers

notimers: red_test_notimers

nocuda: red_test_nocuda

nocuda_notimers: red_test_nocuda_notimers

debug: red_test_DEBUG notimers_DEBUG nocuda_DEBUG nocuda_notimers_DEBUG

notimers_DEBUG: red_test_notimers_DEBUG

nocuda_DEBUG: red_test_nocuda_DEBUG

nocuda_notimers_DEBUG: red_test_nocuda_notimers_DEBUG

red_test_nocuda_notimers: libafosr_cfd_nocuda_notimers.so libafosr_cfd_opencl_core.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -L ./ -lafosr_cfd_nocuda_notimers -lafosr_cfd_opencl_core -o red_test_nocuda_notimers

red_test_nocuda: libafosr_cfd_nocuda.so libafosr_cfd_opencl_core.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -L ./ -lafosr_cfd_nocuda -lafosr_cfd_opencl_core -o red_test_nocuda

red_test_notimers: libafosr_cfd_notimers.so libafosr_cfd_opencl_core.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -L ./ -lafosr_cfd_notimers -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -o red_test_notimers

red_test: libafosr_cfd.so libafosr_cfd_opencl_core.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -L ./ -lafosr_cfd -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -o red_test

libafosr_cfd_nocuda_notimers.so: libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -lafosr_cfd_opencl_core -lOpenCL -o libafosr_cfd_nocuda_notimers.so

libafosr_cfd_nocuda.so: libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -I ./ -L ./ -lafosr_cfd_opencl_core -lOpenCL -o libafosr_cfd_nocuda.so

libafosr_cfd_notimers.so: libafosr_cfd_cuda_core.so libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd_notimers.so

libafosr_cfd.so: libafosr_cfd_cuda_core.so libafosr_cfd_opencl_core.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -I ./ -L ./ -lafosr_cfd_opencl_core -lafosr_cfd_cuda_core -lOpenCL -lcudart -o libafosr_cfd.so

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libafosr_cfd_cuda_core.so:
	nvcc afosr_cfd_cuda_core.cu -Xcompiler -fPIC -shared -I ./ -lcudart  -o libafosr_cfd_cuda_core.so -D __CUDACC__ -arch sm_35

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libafosr_cfd_opencl_core.so:
	g++ afosr_cfd_opencl_core.cpp -fPIC -shared -I ./ -lOpenCL -o libafosr_cfd_opencl_core.so -D __OPENCLCC__

#Debug variants
red_test_nocuda_notimers_DEBUG: libafosr_cfd_nocuda_notimers_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -L ./ -lafosr_cfd_nocuda_notimers_DEBUG -lafosr_cfd_opencl_core_DEBUG -o red_test_nocuda_notimers_DEBUG -g -O0

red_test_nocuda_DEBUG: libafosr_cfd_nocuda_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -L ./ -lafosr_cfd_nocuda_DEBUG -lafosr_cfd_opencl_core_DEBUG -o red_test_nocuda_DEBUG -g -O0

red_test_notimers_DEBUG: libafosr_cfd_notimers_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -L ./ -lafosr_cfd_notimers_DEBUG -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -o red_test_notimers_DEBUG -g -O0

red_test_DEBUG: libafosr_cfd_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so
	gcc red_bootstrapper.c -I ./ -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -L ./ -lafosr_cfd_DEBUG -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -o red_test_DEBUG -g -O0

libafosr_cfd_nocuda_notimers_DEBUG.so: libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -lafosr_cfd_opencl_core_DEBUG -lOpenCL -o libafosr_cfd_nocuda_notimers_DEBUG.so -g -O0

libafosr_cfd_nocuda_DEBUG.so: libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c -fPIC -shared -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -I ./ -L ./ -lafosr_cfd_opencl_core_DEBUG -lOpenCL -o libafosr_cfd_nocuda_DEBUG.so -g -O0

libafosr_cfd_notimers_DEBUG.so: libafosr_cfd_cuda_core_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c  -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -I ./ -L ./ -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -lOpenCL -lcudart -o libafosr_cfd_notimers_DEBUG.so -g -O0

libafosr_cfd_DEBUG.so: libafosr_cfd_cuda_core_DEBUG.so libafosr_cfd_opencl_core_DEBUG.so afosr_cfd.c
	gcc afosr_cfd.c afosr_cfd_timers.c -fPIC -shared -D WITH_CUDA -D WITH_OPENCL -D WITH_OPENMP -D WITH_TIMERS -I ./ -L ./ -lafosr_cfd_opencl_core_DEBUG -lafosr_cfd_cuda_core_DEBUG -lOpenCL -lcudart -o libafosr_cfd_DEBUG.so -g -O0

#I can only get it to provide correct results with the debugging symbols..
#  Something to do with volatiles and sync on the shared memory regions
libafosr_cfd_cuda_core_DEBUG.so:
	nvcc afosr_cfd_cuda_core.cu -Xcompiler -fPIC -shared -I ./ -lcudart  -o libafosr_cfd_cuda_core_DEBUG.so -D __CUDACC__ -arch sm_35 -g -O0

#Stub for once we implement OpenCL
# should include compiling as well as moving the kernel file..
#TODO do something smart with required kernel files..
libafosr_cfd_opencl_core_DEBUG.so:
	g++ afosr_cfd_opencl_core.cpp -fPIC -shared -I ./ -lOpenCL -o libafosr_cfd_opencl_core_DEBUG.so -D __OPENCLCC__ -g -O0
	

clean:
	rm libafosr_cfd_opencl_core.so libafosr_cfd_cuda_core.so libafosr_cfd.so red_test red_test_nocuda red_test_notimers red_test_nocuda_notimers libafosr_cfd_opencl_core_DEBUG.so libafosr_cfd_cuda_core_DEBUG.so libafosr_cfd_DEBUG.so red_test_DEBUG red_test_nocuda_DEBUG red_test_notimers_DEBUG red_test_nocuda_notimers_DEBUG
