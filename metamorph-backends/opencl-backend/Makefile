CL_DEFS = -D __OPENCLCC__ -D WITH_OPENCL
ifeq ($(USE_FPGA),INTEL)
ifeq ($(OPENCL_SINGLE_KERNEL_PROGS),TRUE)
#TODO add aocx build commands and dependencies for extra kernels
metamorph_opencl_intelfpga_reduce_db.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_reduce_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_REDUCE

metamorph_opencl_intelfpga_reduce_fl.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_reduce_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_REDUCE

metamorph_opencl_intelfpga_reduce_ul.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_reduce_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_REDUCE

metamorph_opencl_intelfpga_reduce_in.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_reduce_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_REDUCE

metamorph_opencl_intelfpga_reduce_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_reduce_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_REDUCE
FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_reduce_db.aocx metamorph_opencl_intelfpga_reduce_fl.aocx metamorph_opencl_intelfpga_reduce_ul.aocx metamorph_opencl_intelfpga_reduce_in.aocx metamorph_opencl_intelfpga_reduce_ui.aocx

metamorph_opencl_intelfpga_dotProd_db.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_dotProd_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_DOT_PROD

metamorph_opencl_intelfpga_dotProd_fl.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_dotProd_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_DOT_PROD

metamorph_opencl_intelfpga_dotProd_ul.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_dotProd_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_DOT_PROD

metamorph_opencl_intelfpga_dotProd_in.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_dotProd_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_DOT_PROD

metamorph_opencl_intelfpga_dotProd_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_dotProd_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_DOT_PROD
FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_dotProd_db.aocx metamorph_opencl_intelfpga_dotProd_fl.aocx metamorph_opencl_intelfpga_dotProd_ul.aocx metamorph_opencl_intelfpga_dotProd_in.aocx metamorph_opencl_intelfpga_dotProd_ui.aocx

metamorph_opencl_intelfpga_transpose_2d_face_db.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_transpose_2d_face_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_TRANSPOSE

metamorph_opencl_intelfpga_transpose_2d_face_fl.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_transpose_2d_face_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_TRANSPOSE

metamorph_opencl_intelfpga_transpose_2d_face_ul.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_transpose_2d_face_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_TRANSPOSE

metamorph_opencl_intelfpga_transpose_2d_face_in.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_transpose_2d_face_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_TRANSPOSE

metamorph_opencl_intelfpga_transpose_2d_face_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_transpose_2d_face_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_TRANSPOSE
FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_transpose_2d_face_db.aocx metamorph_opencl_intelfpga_transpose_2d_face_fl.aocx metamorph_opencl_intelfpga_transpose_2d_face_ul.aocx metamorph_opencl_intelfpga_transpose_2d_face_in.aocx metamorph_opencl_intelfpga_transpose_2d_face_ui.aocx

metamorph_opencl_intelfpga_pack_2d_face_db.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_pack_2d_face_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_PACK

metamorph_opencl_intelfpga_pack_2d_face_fl.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_pack_2d_face_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_PACK

metamorph_opencl_intelfpga_pack_2d_face_ul.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_pack_2d_face_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_PACK

metamorph_opencl_intelfpga_pack_2d_face_in.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_pack_2d_face_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_PACK

metamorph_opencl_intelfpga_pack_2d_face_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_pack_2d_face_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_PACK
FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_pack_2d_face_db.aocx metamorph_opencl_intelfpga_pack_2d_face_fl.aocx metamorph_opencl_intelfpga_pack_2d_face_ul.aocx metamorph_opencl_intelfpga_pack_2d_face_in.aocx metamorph_opencl_intelfpga_pack_2d_face_ui.aocx

metamorph_opencl_intelfpga_unpack_2d_face_db.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_unpack_2d_face_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_UNPACK

metamorph_opencl_intelfpga_unpack_2d_face_fl.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_unpack_2d_face_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_UNPACK

metamorph_opencl_intelfpga_unpack_2d_face_ul.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_unpack_2d_face_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_UNPACK

metamorph_opencl_intelfpga_unpack_2d_face_in.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_unpack_2d_face_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_UNPACK

metamorph_opencl_intelfpga_unpack_2d_face_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_unpack_2d_face_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_UNPACK
FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_unpack_2d_face_db.aocx metamorph_opencl_intelfpga_unpack_2d_face_fl.aocx metamorph_opencl_intelfpga_unpack_2d_face_ul.aocx metamorph_opencl_intelfpga_unpack_2d_face_in.aocx metamorph_opencl_intelfpga_unpack_2d_face_ui.aocx

metamorph_opencl_intelfpga_stencil_3d7p_db.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_stencil_3d7p_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_STENCIL

metamorph_opencl_intelfpga_stencil_3d7p_fl.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_stencil_3d7p_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_STENCIL

metamorph_opencl_intelfpga_stencil_3d7p_ul.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_stencil_3d7p_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_STENCIL

metamorph_opencl_intelfpga_stencil_3d7p_in.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_stencil_3d7p_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_STENCIL

metamorph_opencl_intelfpga_stencil_3d7p_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_stencil_3d7p_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_STENCIL
FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_stencil_3d7p_db.aocx metamorph_opencl_intelfpga_stencil_3d7p_fl.aocx metamorph_opencl_intelfpga_stencil_3d7p_ul.aocx metamorph_opencl_intelfpga_stencil_3d7p_in.aocx metamorph_opencl_intelfpga_stencil_3d7p_ui.aocx

metamorph_opencl_intelfpga_csr_db.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_csr_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_CSR

metamorph_opencl_intelfpga_csr_fl.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_csr_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_CSR

metamorph_opencl_intelfpga_csr_ul.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_csr_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_CSR

metamorph_opencl_intelfpga_csr_in.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_csr_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_CSR

metamorph_opencl_intelfpga_csr_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_csr_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CSR
FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_csr_db.aocx metamorph_opencl_intelfpga_csr_fl.aocx metamorph_opencl_intelfpga_csr_ul.aocx metamorph_opencl_intelfpga_csr_in.aocx metamorph_opencl_intelfpga_csr_ui.aocx


metamorph_opencl_intelfpga_crc_ui.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga_crc_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CRC

FPGA_DEPS := $(FPGA_DEPS) metamorph_opencl_intelfpga_crc_ui.aocx
FPGA_DEF := -D OPENCL_SINGLE_KERNEL_PROGS $(FPGA_DEF)
else
FPGA_DEPS = metamorph_opencl_intelfpga.aocx
endif
endif
ifeq ($(OPENCL_SINGLE_KERNEL_PROGS),TRUE)
 CL_DEFS := $(CL_DEFS) -D OPENCL_SINGLE_KERNEL_PROGS
endif

libmetamorph_opencl.so:
	g++ metamorph_opencl.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmetamorph_opencl.so $(CL_DEFS) $(OPENCL_FLAGS) -shared -Wl,-soname,libmetamorph_opencl.so -DMETAMORPH_OCL_KERNEL_PATH=\"$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR):$(MM_LIB):.\"

libmetamorph_opencl_intelfpga.so: $(FPGA_DEPS)
	g++ metamorph_opencl.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) $(FPGA_DEF) $(AOCL_COMPILE_CONFIG) $(FPGA_LIB) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmetamorph_opencl_intelfpga.so -D __OPENCLCC__ $(OPENCL_FLAGS) -shared -Wl,-soname,libmetamorph_opencl_intelfpga.so -DMETAMORPH_OCL_KERNEL_PATH=\"$(shell pwd | sed 's/ /\\ /g'):$(MM_LIB):.\"

metamorph_opencl_intelfpga.aocx: metamorph_opencl.cl
	aoc $(AOC_DEF) metamorph_opencl.cl -o $(MM_LIB)/metamorph_opencl_intelfpga.aocx $(FPGA_DEF)
