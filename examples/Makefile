#CC=gcc
#CC=icc
#USE_MPI=TRUE

DEPS =
ifeq ($(USE_CUDA),TRUE)
BUILD_FLAGS += -D WITH_CUDA -lmetamorph_cuda -L$(CUDA_LIB_DIR) -lcudart
endif
ifeq ($(USE_OPENCL),TRUE)
BUILD_FLAGS += -D WITH_OPENCL -lmetamorph_opencl -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL
DEPS += metamorph_opencl.cl
endif
ifeq ($(USE_OPENMP),TRUE)
BUILD_FLAGS += -D WITH_OPENMP
ifeq ($(USE_MIC),TRUE)
BUILD_FLAGS += -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB) -mmic -lmetamorph_openmp_mic
else
BUILD_FLAGS += -lmetamorph_openmp
endif
endif
ifeq ($(USE_TIMERS),TRUE)
BUILD_FLAGS += -D WITH_TIMERS
endif
ifeq ($(USE_MPI),TRUE)
BUILD_FLAGS += -D WITH_MPI -lmetamorph_mpi -L$(MPI_DIR)/lib -I$(MPI_DIR)/include
CC=$(MPICC)
endif

all: torus_reduce_test csr_alt crc_alt

.PHONY: torus_reduce_test torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl
torus_reduce_test: torus_reduce_meta.c $(DEPS)
ifneq ($(USE_TIMERS),TRUE)
$(error torus_reduce_test requires Timers)
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -lmetamorph -lmetamorph_profiling $(BUILD_FLAGS) -o torus_reduce_test
endif
#torus_reduce_test: torus_reduce_meta.c
#ifeq ($(USE_MPI),TRUE)
#	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_CUDA -D WITH_OPENMP -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -lmetamorph -o torus_reduce_test
#else
#	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_CUDA -D WITH_OPENMP -D WITH_OPENCL -D WITH_TIMERS -lmetamorph -o torus_reduce_test
#endif

torus_reduce_test_mp: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPI_DIR)/include -L $(MPI_DIR)/lib -D $(G_TYPE) -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmetamorph_mp -lmetamorph_openmp -o torus_reduce_test_mp
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_OPENMP -D WITH_TIMERS -lmetamorph_mp -lmetamorph_openmp -o torus_reduce_test_mp
endif

torus_reduce_test_mic: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=icc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)-mic/include -L $(MPICH_DIR)-mic/lib -D $(G_TYPE) -mmic -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmetamorph_mic -lmetamorph_openmp_mic -o torus_reduce_test_mic -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB)
else
	icc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -mmic -D WITH_OPENMP -D WITH_TIMERS -lmetamorph_mic -libmetamorph_openmp_mic -o torus_reduce_test_mic -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB)
endif

torus_reduce_test_cu: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -lmetamorph_cu -lmetamorph_cuda -o torus_reduce_test_cu
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_CUDA -D WITH_TIMERS -lmetamorph_cu -o torus_reduce_test_cu
endif
	
torus_reduce_test_cl: torus_reduce_meta.c | metamorph_opencl.cl
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -lmetamorph_cl -lmetamorph_opencl -lOpenCL -o torus_reduce_test_cl
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -L /home/jehandad/rte/opt/altera/aocl-rte/host/linux64/lib/ -D $(G_TYPE) -D WITH_OPENCL -D WITH_TIMERS -lmetamorph_cl -lOpenCL -o torus_reduce_test_cl
endif

metamorph_opencl.cl:
	ln -s $(MM_CL)/metamorph_opencl.cl metamorph_opencl.cl

reduce_alt: reduce_alt.c | metamorph_opencl_intelfpga.cl
	$(CC) reduce_alt.c $(OPT_LVL) $(INCLUDES) $(FPGA_DEF) -L $(MM_LIB) $(AOCL_LINK_CONFIG) $(FPGA_LIB) -D $(G_TYPE) -lmetamorph -lmetamorph_opencl_intelfpga -lOpenCL -o reduce_alt

#file never added to repo
#csr_alt: csr_alt.c | metamorph_opencl_intelfpga.cl
#	$(CC) csr_alt.c $(OPT_LVL) $(INCLUDES) $(FPGA_DEF) -L $(MM_LIB) $(AOCL_LINK_CONFIG) $(FPGA_LIB) -D $(G_TYPE) -lmetamorph -lmetamorph_opencl_intelfpga -lOpenCL -o csr_alt

#file never added to repo
#crc_alt: crc_alt.c | metamorph_opencl_intelfpga.cl
#	$(CC) crc_alt.c $(OPT_LVL) $(INCLUDES) $(FPGA_DEF) -L $(MM_LIB) $(AOCL_LINK_CONFIG) $(FPGA_LIB) -D $(G_TYPE) -lmetamorph -lmetamorph_opencl_intelfpga -lOpenCL -o crc_alt


metamorph_opencl_intelfpga.cl:
	ln -s $(MM_DIR)/bin/metamorph_opencl_intelfpga.aocx metamorph_opencl_intelfpga.aocx
