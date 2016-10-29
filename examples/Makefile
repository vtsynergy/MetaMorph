CC=gcc
#CC=icc
USE_MPI=TRUE

.PHONY: torus_reduce_test torus_reduce_test_mp torus_reduce_mic torus_reduce_test_cu torus_reduce_test_cl
torus_reduce_test: torus_reduce_meta.c 
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_CUDA -D WITH_OPENMP -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -lmetamorph -o torus_reduce_test
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_CUDA -D WITH_OPENMP -D WITH_OPENCL -D WITH_TIMERS -lmetamorph -o torus_reduce_test
endif

torus_reduce_test_mp: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmetamorph_mp -lmm_openmp_backend -o torus_reduce_test_mp
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_OPENMP -D WITH_TIMERS -lmetamorph_mp -lmm_openmp_backend -o torus_reduce_test_mp
endif

torus_reduce_mic: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=icc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)-mic/include -L $(MPICH_DIR)-mic/lib -D $(G_TYPE) -mmic -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmetamorph_mic -libmm_openmp_backend_mic -o torus_reduce_test_mic -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB)
else
	icc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -mmic -D WITH_OPENMP -D WITH_TIMERS -lmetamorph_mic -libmm_openmp_backend_mic -o torus_reduce_test_mic -Wl,-rpath,/opt/intel/lib/mic,-rpath,$(MM_LIB)
endif

torus_reduce_test_cu: torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -lmetamorph_cu -lmm_cuda_backend -o torus_reduce_test_cu
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_CUDA -D WITH_TIMERS -lmetamorph_cu -o torus_reduce_test_cu
endif
	
torus_reduce_test_cl: cl_link torus_reduce_meta.c
ifeq ($(USE_MPI),TRUE)
	mpicc torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -D $(G_TYPE) -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -lmetamorph_cl -lmm_opencl_backend -lOpenCL -o torus_reduce_test_cl
else
	$(CC) torus_reduce_meta.c $(OPT_LVL) $(INCLUDES) -L $(MM_LIB) -D $(G_TYPE) -D WITH_OPENCL -D WITH_TIMERS -lmetamorph_cl -lOpenCL -o torus_reduce_test_cl
endif

cl_link:
	ln -s $(MM_CL)/mm_opencl_backend.cl mm_opencl_backend.cl