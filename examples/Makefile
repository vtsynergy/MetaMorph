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
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -lmetamorph -lmetamorph_profiling $(BUILD_FLAGS) -o torus_reduce_test

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
