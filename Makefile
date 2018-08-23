#root directories 
export MPICH_DIR =/home/ammhelal/MPICH-3.2/install
export MM_DIR=/home/ammhelal/metamorph-public

export MM_CORE=$(MM_DIR)/metamorph-core
export MM_MP=$(MM_DIR)/metamorph-backends/openmp-backend
export MM_CU=$(MM_DIR)/metamorph-backends/cuda-backend
export MM_CL=$(MM_DIR)/metamorph-backends/opencl-backend
export MM_EX=$(MM_DIR)/examples

export MM_LIB=$(MM_DIR)/lib

INCLUDES  = -I $(MM_DIR)/include
INCLUDES += -I$(MM_MP)
INCLUDES += -I$(MM_CU)
INCLUDES += -I$(MM_CL)

export INCLUDES

export G_TYPE = DOUBLE
#export OPT_LVL = -g -DDEBUG
export OPT_LVL = -O3
export L_FLAGS= -fPIC -shared

CC=gcc
#CC=icc
USE_MPI=TRUE
#USE_MPI=FALSE


ifeq ($(CC),gcc)
CC_FLAGS = $(OPT_LVL) $(L_FLAGS) -fopenmp
else
CC_FLAGS= $(OPT_LVL) $(L_FLAGS) -openmp
endif


.PHONY: metamorph_all examples
metamorph_all: libmetamorph.so libmetamorph_mp.so libmetamorph_mic.so libmetamorph_cl.so libmetamorph_cu.so

libmetamorph.so: libmm_openmp_backend.so libmm_cuda_backend.so libmm_opencl_backend.so
ifeq ($(USE_MPI),TRUE)
	mpicc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_OPENCL -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -L /usr/local/cuda/lib64 -lmm_openmp_backend  -lmm_cuda_backend -lmm_opencl_backend -lOpenCL -lcudart -o $(MM_LIB)/libmetamorph.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_OPENCL -D WITH_CUDA -D WITH_TIMERS -L /usr/local/cuda/lib64 -lmm_openmp_backend  -lmm_cuda_backend -lmm_opencl_backend -lOpenCL -lcudart -o $(MM_LIB)/libmetamorph.so
endif

libmetamorph_mp.so: libmm_openmp_backend.so
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -I $(MPICH_DIR)/include -L $(MM_LIB) -L $(MPICH_DIR)/lib -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI  -lmm_openmp_backend -o $(MM_LIB)/libmetamorph_mp.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -lmm_openmp_backend -o $(MM_LIB)/libmetamorph_mp.so
endif

libmetamorph_mic.so: libmm_openmp_backend_mic.so
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=icc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) -openmp -mmic $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmm_openmp_backend_mic -I $(MPICH_DIR)-mic/include -L $(MPICH_DIR)-mic/lib -o $(MM_LIB)/libmetamorph_mic.so
else
	icc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) -openmp -mmic $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -lmm_openmp_backend_mic -o $(MM_LIB)/libmetamorph_mic.so
endif

libmetamorph_cu.so: libmm_cuda_backend.so
ifeq ($(USE_MPI),TRUE)
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -L /usr/local/cuda/lib64 -lmm_cuda_backend -lcudart -o $(MM_LIB)/libmetamorph_cu.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_CUDA -D WITH_TIMERS -L /usr/local/cuda/lib64 -lmm_cuda_backend -lcudart -o $(MM_LIB)/libmetamorph_cu.so
endif
	
libmetamorph_cl.so: libmm_opencl_backend.so
ifeq ($(USE_MPI),TRUE)
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -lmm_opencl_backend -lOpenCL -o $(MM_LIB)/libmetamorph_cl.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENCL -D WITH_TIMERS  -lmm_opencl_backend -lOpenCL -o $(MM_LIB)/libmetamorph_cl.so
endif

libmm_openmp_backend.so:	
	cd $(MM_MP) && $(MAKE) libmm_openmp_backend.so $(MFLAGS)
	
libmm_openmp_backend_mic.so:
	cd $(MM_MP) && $(MAKE) libmm_openmp_backend_mic.so $(MFLAGS)

libmm_cuda_backend.so:
	cd $(MM_CU) && $(MAKE) libmm_cuda_backend.so $(MFLAGS)

libmm_opencl_backend.so:
	cd $(MM_CL) && $(MAKE) libmm_opencl_backend.so $(MFLAGS)

examples: 
	cd $(MM_EX) && $(MAKE) torus_reduce_test $(MFLAGS)
#	cd $(MM_EX) && $(MAKE) torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl $(MFLAGS)
	
clean:
	rm $(MM_LIB)/libmetamorph*.so $(MM_LIB)/libmm*.so
