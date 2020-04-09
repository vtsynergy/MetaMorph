metamorph_cuda.o: metamorph_cuda.cu
	$(NVCC) metamorph_cuda.cu $(OPT_LVL) -c -Xcompiler -fPIC $(INCLUDES) -o metamorph_cuda.o

libmetamorph_cuda.so: metamorph_cuda.o
	$(CC) metamorph_cuda.o $(OPT_LVL) $(L_FLAGS) $(INCLUDES) -L$(CUDA_LIB_DIR) -lcudart -o $(MM_LIB)/libmetamorph_cuda.so -Wl,-soname,libmetamorph_cuda.so
