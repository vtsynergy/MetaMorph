#CC=gcc
#CC=icc
#USE_MPI=TRUE

DEPS =
ifeq ($(USE_CUDA),TRUE)
BUILD_FLAGS += -D WITH_CUDA -lmm_cuda_backend -L$(CUDA_LIB_DIR) -lcudart
endif
ifeq ($(USE_OPENCL),TRUE)
BUILD_FLAGS += -D WITH_OPENCL -lmm_opencl_backend -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL
DEPS += mm_opencl_backend.cl
endif
ifeq ($(USE_OPENMP),TRUE)
BUILD_FLAGS += -D WITH_OPENMP
ifeq ($(USE_MIC),TRUE)
BUILD_FLAGS += -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB) -mmic -lmm_openmp_mic_backend
else
BUILD_FLAGS += -lmm_openmp_backend
endif
endif
ifeq ($(USE_TIMERS),TRUE)
BUILD_FLAGS += -D WITH_TIMERS -lmm_profiling
endif
ifeq ($(USE_MPI),TRUE)
BUILD_FLAGS += -D WITH_MPI -lmm_mpi -L$(MPI_DIR)/lib -I$(MPI_DIR)/include
CC=$(MPICC)
endif

all: torus_reduce_test csr_alt crc_alt

.PHONY: torus_reduce_test torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl
torus_reduce_test: torus_reduce_meta.c $(DEPS)
ifneq ($(USE_TIMERS),TRUE)
$(error torus_reduce_test requires Timers)
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -lmetamorph $(BUILD_FLAGS) -o torus_reduce_test
endif
#torus_reduce_test: torus_reduce_meta.c
#ifeq ($(USE_MPI),TRUE)
#	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_CUDA -D WITH_OPENMP -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -lmetamorph -o torus_reduce_test
#else
#	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_CUDA -D WITH_OPENMP -D WITH_OPENCL -D WITH_TIMERS -lmetamorph -o torus_reduce_test
#endif

torus_reduce_test_mp: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPI_DIR)/include -L $(MPI_DIR)/lib -D $(G_TYPE) -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmetamorph_mp -lmm_openmp_backend -o torus_reduce_test_mp
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_OPENMP -D WITH_TIMERS -lmetamorph_mp -lmm_openmp_backend -o torus_reduce_test_mp
endif

torus_reduce_test_mic: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=icc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)-mic/include -L $(MPICH_DIR)-mic/lib -D $(G_TYPE) -mmic -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmetamorph_mic -lmm_openmp_backend_mic -o torus_reduce_test_mic -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB)
else
	icc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -mmic -D WITH_OPENMP -D WITH_TIMERS -lmetamorph_mic -libmm_openmp_backend_mic -o torus_reduce_test_mic -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB)
endif

torus_reduce_test_cu: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -lmetamorph_cu -lmm_cuda_backend -o torus_reduce_test_cu
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_CUDA -D WITH_TIMERS -lmetamorph_cu -o torus_reduce_test_cu
endif
	
torus_reduce_test_cl: torus_reduce_meta.c | mm_opencl_backend.cl
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -lmetamorph_cl -lmm_opencl_backend -lOpenCL -o torus_reduce_test_cl
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -L /home/jehandad/rte/opt/altera/aocl-rte/host/linux64/lib/ -D $(G_TYPE) -D WITH_OPENCL -D WITH_TIMERS -lmetamorph_cl -lOpenCL -o torus_reduce_test_cl
endif

mm_opencl_backend.cl:
	ln -s $(MM_CL)/mm_opencl_backend.cl mm_opencl_backend.cl

reduce_alt: reduce_alt.c | mm_opencl_intelfpga_backend.cl
	$(CC) reduce_alt.c $(OPT_LVL) $(INCLUDES) $(FPGA_DEF) -L $(MM_LIB) $(AOCL_LINK_CONFIG) $(FPGA_LIB) -D $(G_TYPE) -lmetamorph -lmm_opencl_intelfpga_backend -lOpenCL -o reduce_alt

#file never added to repo
#csr_alt: csr_alt.c | mm_opencl_intelfpga_backend.cl
#	$(CC) csr_alt.c $(OPT_LVL) $(INCLUDES) $(FPGA_DEF) -L $(MM_LIB) $(AOCL_LINK_CONFIG) $(FPGA_LIB) -D $(G_TYPE) -lmetamorph -lmm_opencl_intelfpga_backend -lOpenCL -o csr_alt

#file never added to repo
#crc_alt: crc_alt.c | mm_opencl_intelfpga_backend.cl
#	$(CC) crc_alt.c $(OPT_LVL) $(INCLUDES) $(FPGA_DEF) -L $(MM_LIB) $(AOCL_LINK_CONFIG) $(FPGA_LIB) -D $(G_TYPE) -lmetamorph -lmm_opencl_intelfpga_backend -lOpenCL -o crc_alt


mm_opencl_intelfpga_backend.cl:
	ln -s $(MM_DIR)/bin/mm_opencl_intelfpga_backend.aocx mm_opencl_intelfpga_backend.aocx
