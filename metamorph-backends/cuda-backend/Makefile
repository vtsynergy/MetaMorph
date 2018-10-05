mm_cuda_backend.o:
	$(NVCC) mm_cuda_backend.cu $(OPT_LVL) -Xcompiler $(L_FLAGS) $(INCLUDES) -o mm_cuda_backend.o -D __CUDACC__ -arch sm_20 -shared

libmm_cuda_backend.so: mm_cuda_backend.o
	$(CC) mm_cuda_backend.o $(OPT_LVL) $(L_FLAGS) -L$(CUDA_LIB_DIR) -lcudart -o $(MM_LIB)/libmm_cuda_backend.so -shared -Wl,-soname,libmm_cuda_backend.so
