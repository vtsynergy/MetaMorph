
libmm_cuda_backend.so:
	$(NVCC) mm_cuda_backend.cu $(OPT_LVL) -Xcompiler $(L_FLAGS) $(INCLUDES) -L$(CUDA_LIB_DIR) -lcudart -o $(MM_LIB)/libmm_cuda_backend.so -D __CUDACC__ -arch sm_20 -shared -Wl,-soname,libmm_cuda_backend.so
