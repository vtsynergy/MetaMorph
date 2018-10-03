#root directories 
export MPICH_DIR =/home/ammhelal/MPICH-3.2/install
export MM_DIR=/media/hdd/mwasfy/Metamorph

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

export G_TYPE = UNSIGNED_INTEGER
#export OPT_LVL = -g -DDEBUG
export OPT_LVL = -O3
export L_FLAGS= -fPIC -shared

CC=gcc
#CC=icc
#USE_MPI=TRUE
USE_MPI=FALSE
#USE_EMULATOR=TRUE
USE_EMULATOR=FALSE

ifeq ($(CC),gcc)
CC_FLAGS = $(OPT_LVL) $(L_FLAGS) -fopenmp
else
CC_FLAGS= $(OPT_LVL) $(L_FLAGS) -openmp
endif

####################### FPGA ##################################
# Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# OpenCL compile and link flags.
export AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
export AOCL_LINK_CONFIG := $(shell aocl link-config )


ifeq ($(USE_EMULATOR),TRUE)
export AOC_DEF= -march=emulator -v --board bdw_fpga_v1.0
else
export AOC_DEF= -v --board bdw_fpga_v1.0
endif

#-D WITH_TIMERS
export FPGA_DEF=-D WITH_OPENCL -D __FPGA__ -D WITH_TIMERS -D KERNEL_CRC -D FPGA_UNSIGNED_INTEGER
#export FPGA_DEF=-D WITH_OPENCL -D WITH_TIMERS -D __FPGA__ -D KERNEL_STENCIL -D FPGA_DOUBLE
#export FPGA_LIB=/home/jehandad/rte/opt/altera/aocl-rte/host/linux64/lib/
export FPGA_LIB=-L /media/hdd/jehandad/altera_pro/16.0/hld/host/linux64/lib -L $(AALSDK)/lib

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
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(AOCL_COMPILE_CONFIG) $(INCLUDES) -L $(MM_LIB) $(FPGA_LIB) -D WITH_OPENCL -D WITH_TIMERS -lmm_opencl_backend -lOpenCL -o $(MM_LIB)/libmetamorph_cl.so
endif

#/media/hdd/jehandad/altera_pro/16.0/hld/host/linux64/lib 
libmm_openmp_backend.so:	
	cd $(MM_MP) && $(MAKE) libmm_openmp_backend.so $(MFLAGS)
	
libmm_openmp_backend_mic.so:
	cd $(MM_MP) && $(MAKE) libmm_openmp_backend_mic.so $(MFLAGS)

libmm_cuda_backend.so:
	cd $(MM_CU) && $(MAKE) libmm_cuda_backend.so $(MFLAGS)

libmm_opencl_backend.so:
	cd $(MM_CL) && $(MAKE) libmm_opencl_backend.so $(MFLAGS)

examples: 
	cd $(MM_EX) && $(MAKE) csr_alt $(MFLAGS)
#	cd $(MM_EX) && $(MAKE) torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl $(MFLAGS)

crc_ex:
	cd $(MM_EX) && $(MAKE) crc_alt $(MFLAGS)	
clean:
	rm $(MM_LIB)/libmetamorph*.so $(MM_LIB)/libmm*.so

refresh:
	rm $(MM_EX)/crc_alt $(MM_EX)/mm_opencl_backend_alt.aocx
