
libmm_cuda_backend.so:
	nvcc mm_cuda_backend.cu $(OPT_LVL) -Xcompiler $(L_FLAGS) $(INCLUDES) -L /usr/local/cuda/lib64 -lcudart -o $(MM_LIB)/libmm_cuda_backend.so -D __CUDACC__ -arch sm_20