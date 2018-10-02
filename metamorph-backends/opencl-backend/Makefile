
libmm_opencl_backend.so:
	g++ mm_opencl_backend.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmm_opencl_backend.so -D __OPENCLCC__ -shared -Wl,-soname,libmm_opencl_backend.so
