#CC=gcc
#CC=icc
#USE_MPI=TRUE

BUILD_FLAGS=
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
BUILD_FLAGS += -D WITH_TIMERS
endif
ifeq ($(USE_MPI),TRUE)
BUILD_FLAGS += -D WITH_MPI -L$(MPI_DIR)/lib -I$(MPI_DIR)/include
endif

.PHONY: torus_reduce_test torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl
torus_reduce_test: torus_reduce_meta.c $(DEPS)
ifneq ($(USE_MPI),TRUE)
$(error torus_reduce_test requires MPI)
else ifneq ($(USE_TIMERS),TRUE)
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
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_OPENCL -D WITH_TIMERS -lmetamorph_cl -lOpenCL -o torus_reduce_test_cl
endif

mm_opencl_backend.cl:
	ln -s $(MM_CL)/mm_opencl_backend.cl mm_opencl_backend.cl
