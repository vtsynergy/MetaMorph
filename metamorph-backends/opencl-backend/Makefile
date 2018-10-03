
libmm_opencl_backend.so:
	g++ mm_opencl_backend.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmm_opencl_backend.so -D __OPENCLCC__ -shared -Wl,-soname,libmm_opencl_backend.so

libmm_opencl_intelfpga_backend.so:
	g++ mm_opencl_backend_alt.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) $(FPGA_DEF) $(AOCL_COMPILE_CONFIG) $(FPGA_LIB) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmm_opencl_intelfpga_backend.so -D __OPENCLCC__ -shared -Wl,-soname,libmm_opencl_intelfpga_backend.so

mm_opencl_intelfpga_backend.aocx: mm_opencl_backend_alt.cl
	aoc $(AOC_DEF) mm_opencl_intelfpga_backend.cl -o $(MM_DIR)/bin/mm_opencl_intelfpga_backend.aocx $(FPGA_DEF)
