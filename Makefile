#root directories 
export MPICH_DIR =/home/psath/MPICH-3.1.4/install
export MM_DIR=/home/psath/metamorphWorkspace/metamorph

export MM_CORE=$(MM_DIR)/metamorph-core
export MM_MP=$(MM_DIR)/metamorph-backends/openmp-backend
export MM_CU=$(MM_DIR)/metamorph-backends/cuda-backend
export MM_CL=$(MM_DIR)/metamorph-backends/opencl-backend
export MM_EX=$(MM_DIR)/examples

export MM_LIB=$(MM_DIR)/lib

export MM_GEN_CL=$(MM_DIR)/metamorph-generators/opencl

USE_CUDA=FALSE
CUDA_LIB_DIR=/usr/local/cuda/lib64
USE_OPENCL=TRUE
OPENCL_LIB_DIR=/opt/rocm/opencl/lib/x86_64/
#Set to one of INTEL, XILINX, or blank
USE_FPGA=
MM_AOCX_DIR=$(MM_LIB)
USE_OPENMP=FALSE
USE_MIC=FALSE
ICC_BIN=/opt/intel/bin/icc
USE_MPI=FALSE
USE_TIMERS=TRUE

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


#Configure backends to compile in
MM_DEPS= $(MM_CORE)/metamorph.c
MM_COMPONENTS=
ifeq ($(USE_MPI),TRUE)
MM_COMPONENTS += -D WITH_MPI $(MM_CORE)/metamorph_mpi.c
MM_DEPS += $(MM_CORE)/metamorph_mpi.c
CC = mpicc -cc=$(CC)
endif
ifeq ($(USE_TIMERS),TRUE)
MM_COMPONENTS += -D WITH_TIMERS $(MM_CORE)/metamorph_timers.c
MM_DEPS += $(MM_CORE)/metamorph_timers.c
endif

ifeq ($(USE_CUDA),TRUE)
MM_COMPONENTS += -D WITH_CUDA -lmm_cuda_backend -L$(CUDA_LIB_DIR) -lcudart
MM_DEPS += libmm_cuda_backend.so
endif
ifeq ($(USE_OPENCL),TRUE)
MM_COMPONENTS += -D WITH_OPENCL
ifeq ($(USE_FPGA),INTEL)
MM_COMPONENTS += -D WITH_INTELFPGA -lmm_opencl_intelfpga_backend
MM_DEPS += libmm_opencl_intelfpga_backend.so
else ifeq ($(USE_FPGA),XILINX)
MM_COMPONENTS += -D WITH_XILINXFPGA -lmm_opencl_xilinx_backend
MM_DEPS += libmm_opencl_xilinx_backend.so
else
MM_COMPONENTS += -lmm_opencl_backend
MM_DEPS += libmm_opencl_backend.so
endif
MM_COMPONENTS += -L$(OPENCL_LIB_DIR) -lOpenCL
endif
ifeq ($(USE_OPENMP),TRUE)
MM_COMPONENTS += -D WITH_OPENMP
ifeq ($(USE_MIC),TRUE)
ifeq($(USE_MPI),TRUE)
CC = mpicc -cc=$(ICC_BIN)
else
CC = $(ICC_BIN)
endif
CC_FLAGS += -mmic
MM_COMPONENTS += -lmm_openmp_backend_mic
MM_DEPS += libmm_openmp_mic_backend.so
else
MM_COMPONENTS += -lmm_openmp_backend
MM_DEPS += libmm_openmp_backend.so
endif
ifeq ($(CC),gcc)
CC_FLAGS = $(OPT_LVL) $(L_FLAGS) -fopenmp
else
CC_FLAGS= $(OPT_LVL) $(L_FLAGS) -openmp
endif



#.PHONY: metamorph_all examples
.PHONY: libmetamorph.so examples
#metamorph_all: libmetamorph.so libmetamorph_mp.so libmetamorph_mic.so libmetamorph_cl.so libmetamorph_cu.so

#libmetamorph.so: libmm_openmp_backend.so libmm_cuda_backend.so libmm_opencl_backend.so
#ifeq ($(USE_MPI),TRUE)
#	mpicc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_OPENCL -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -L /usr/local/cuda/lib64 -lmm_openmp_backend  -lmm_cuda_backend -lmm_opencl_backend -lOpenCL -lcudart -o $(MM_LIB)/libmetamorph.so
#else
#	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_OPENCL -D WITH_CUDA -D WITH_TIMERS -L /usr/local/cuda/lib64 -lmm_openmp_backend  -lmm_cuda_backend -lmm_opencl_backend -lOpenCL -lcudart -o $(MM_LIB)/libmetamorph.so
#endif
libmetamorph.so: $(MM_DEPS)
	$(CC) $(MM_CORE)/metamorph.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) $(MM_COMPONENTS) -o $(MM_LIB)/libmetamorph.so

#do we need these old single-backend targets now that the above is modular?
libmetamorph_mp.so: libmm_openmp_backend.so
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -I $(MPICH_DIR)/include -L $(MM_LIB) -L $(MPICH_DIR)/lib -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI  -lmm_openmp_backend -o $(MM_LIB)/libmetamorph_mp.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -lmm_openmp_backend -o $(MM_LIB)/libmetamorph_mp.so
endif

libmetamorph_mic.so: libmm_openmp_mic_backend.so
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=icc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) -openmp -mmic $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmm_openmp_mic_backend -I $(MPICH_DIR)-mic/include -L $(MPICH_DIR)-mic/lib -o $(MM_LIB)/libmetamorph_mic.so
else
	icc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) -openmp -mmic $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -lmm_openmp_mic_backend -o $(MM_LIB)/libmetamorph_mic.so
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
	
libmm_openmp_mic_backend.so:
	cd $(MM_MP) && $(MAKE) libmm_openmp_backend_mic.so $(MFLAGS)

libmm_cuda_backend.so:
	cd $(MM_CU) && $(MAKE) libmm_cuda_backend.so $(MFLAGS)

libmm_opencl_backend.so:
	cd $(MM_CL) && $(MAKE) libmm_opencl_backend.so $(MFLAGS)

examples: 
	cd $(MM_EX) && $(MAKE) torus_reduce_test $(MFLAGS)
#	cd $(MM_EX) && $(MAKE) torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl $(MFLAGS)

generators: MetaGen-CL

MetaGen-CL:
	cd $(MM_GEN_CL) && $(MAKE) metagen-cl $(MFLAGS)
	
clean:
	rm $(MM_LIB)/libmetamorph*.so $(MM_LIB)/libmm*.so
