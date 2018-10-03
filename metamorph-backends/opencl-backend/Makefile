
libmm_opencl_backend.so:
	g++ mm_opencl_backend_alt.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) $(FPGA_DEF) $(AOCL_COMPILE_CONFIG) $(FPGA_LIB) -lOpenCL -o $(MM_LIB)/libmm_opencl_backend.so -D __OPENCLCC__

mm_opencl_backend_alt.aocx: mm_opencl_backend_alt.cl
	aoc $(AOC_DEF) mm_opencl_backend_alt.cl -o $(MM_DIR)/bin/mm_opencl_backend_alt.aocx $(FPGA_DEF)
