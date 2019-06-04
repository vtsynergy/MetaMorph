CL_DEFS = -D __OPENCLCC__ -D WITH_OPENCL
ifeq ($(USE_FPGA),INTEL)
ifeq ($(OPENCL_SINGLE_KERNEL_PROGS),TRUE)
#TODO add aocx build commands and dependencies for extra kernels
mm_opencl_intelfpga_backend_reduce_db.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_reduce_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_REDUCE

mm_opencl_intelfpga_backend_reduce_fl.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_reduce_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_REDUCE

mm_opencl_intelfpga_backend_reduce_ul.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_reduce_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_REDUCE

mm_opencl_intelfpga_backend_reduce_in.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_reduce_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_REDUCE

mm_opencl_intelfpga_backend_reduce_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_reduce_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_REDUCE
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_reduce_db.aocx mm_opencl_intelfpga_backend_reduce_fl.aocx mm_opencl_intelfpga_backend_reduce_ul.aocx mm_opencl_intelfpga_backend_reduce_in.aocx mm_opencl_intelfpga_backend_reduce_ui.aocx

mm_opencl_intelfpga_backend_dotProd_db.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_dotProd_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_DOT_PROD

mm_opencl_intelfpga_backend_dotProd_fl.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_dotProd_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_DOT_PROD

mm_opencl_intelfpga_backend_dotProd_ul.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_dotProd_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_DOT_PROD

mm_opencl_intelfpga_backend_dotProd_in.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_dotProd_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_DOT_PROD

mm_opencl_intelfpga_backend_dotProd_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_dotProd_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_DOT_PROD
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_dotProd_db.aocx mm_opencl_intelfpga_backend_dotProd_fl.aocx mm_opencl_intelfpga_backend_dotProd_ul.aocx mm_opencl_intelfpga_backend_dotProd_in.aocx mm_opencl_intelfpga_backend_dotProd_ui.aocx

mm_opencl_intelfpga_backend_transpose_2d_face_db.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_transpose_2d_face_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_TRANSPOSE

mm_opencl_intelfpga_backend_transpose_2d_face_fl.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_transpose_2d_face_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_TRANSPOSE

mm_opencl_intelfpga_backend_transpose_2d_face_ul.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_transpose_2d_face_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_TRANSPOSE

mm_opencl_intelfpga_backend_transpose_2d_face_in.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_transpose_2d_face_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_TRANSPOSE

mm_opencl_intelfpga_backend_transpose_2d_face_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_transpose_2d_face_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_TRANSPOSE
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_transpose_2d_face_db.aocx mm_opencl_intelfpga_backend_transpose_2d_face_fl.aocx mm_opencl_intelfpga_backend_transpose_2d_face_ul.aocx mm_opencl_intelfpga_backend_transpose_2d_face_in.aocx mm_opencl_intelfpga_backend_transpose_2d_face_ui.aocx

mm_opencl_intelfpga_backend_pack_2d_face_db.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_pack_2d_face_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_PACK

mm_opencl_intelfpga_backend_pack_2d_face_fl.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_pack_2d_face_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_PACK

mm_opencl_intelfpga_backend_pack_2d_face_ul.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_pack_2d_face_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_PACK

mm_opencl_intelfpga_backend_pack_2d_face_in.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_pack_2d_face_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_PACK

mm_opencl_intelfpga_backend_pack_2d_face_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_pack_2d_face_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_PACK
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_pack_2d_face_db.aocx mm_opencl_intelfpga_backend_pack_2d_face_fl.aocx mm_opencl_intelfpga_backend_pack_2d_face_ul.aocx mm_opencl_intelfpga_backend_pack_2d_face_in.aocx mm_opencl_intelfpga_backend_pack_2d_face_ui.aocx

mm_opencl_intelfpga_backend_unpack_2d_face_db.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_unpack_2d_face_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_UNPACK

mm_opencl_intelfpga_backend_unpack_2d_face_fl.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_unpack_2d_face_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_UNPACK

mm_opencl_intelfpga_backend_unpack_2d_face_ul.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_unpack_2d_face_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_UNPACK

mm_opencl_intelfpga_backend_unpack_2d_face_in.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_unpack_2d_face_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_UNPACK

mm_opencl_intelfpga_backend_unpack_2d_face_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_unpack_2d_face_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_UNPACK
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_unpack_2d_face_db.aocx mm_opencl_intelfpga_backend_unpack_2d_face_fl.aocx mm_opencl_intelfpga_backend_unpack_2d_face_ul.aocx mm_opencl_intelfpga_backend_unpack_2d_face_in.aocx mm_opencl_intelfpga_backend_unpack_2d_face_ui.aocx

mm_opencl_intelfpga_backend_stencil_3d7p_db.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_stencil_3d7p_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_STENCIL

mm_opencl_intelfpga_backend_stencil_3d7p_fl.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_stencil_3d7p_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_STENCIL

mm_opencl_intelfpga_backend_stencil_3d7p_ul.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_stencil_3d7p_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_STENCIL

mm_opencl_intelfpga_backend_stencil_3d7p_in.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_stencil_3d7p_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_STENCIL

mm_opencl_intelfpga_backend_stencil_3d7p_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_stencil_3d7p_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_STENCIL
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_stencil_3d7p_db.aocx mm_opencl_intelfpga_backend_stencil_3d7p_fl.aocx mm_opencl_intelfpga_backend_stencil_3d7p_ul.aocx mm_opencl_intelfpga_backend_stencil_3d7p_in.aocx mm_opencl_intelfpga_backend_stencil_3d7p_ui.aocx

mm_opencl_intelfpga_backend_csr_db.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_csr_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_CSR

mm_opencl_intelfpga_backend_csr_fl.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_csr_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_CSR

mm_opencl_intelfpga_backend_csr_ul.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_csr_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_CSR

mm_opencl_intelfpga_backend_csr_in.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_csr_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_CSR

mm_opencl_intelfpga_backend_csr_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_csr_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CSR
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_csr_db.aocx mm_opencl_intelfpga_backend_csr_fl.aocx mm_opencl_intelfpga_backend_csr_ul.aocx mm_opencl_intelfpga_backend_csr_in.aocx mm_opencl_intelfpga_backend_csr_ui.aocx

#mm_opencl_intelfpga_backend_crc_db.aocx: mm_opencl_backend.cl
#	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_crc_db.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_CRC

#mm_opencl_intelfpga_backend_crc_fl.aocx: mm_opencl_backend.cl
#	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_crc_fl.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_CRC

#mm_opencl_intelfpga_backend_crc_ul.aocx: mm_opencl_backend.cl
#	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_crc_ul.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_CRC

#mm_opencl_intelfpga_backend_crc_in.aocx: mm_opencl_backend.cl
#	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_crc_in.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_CRC

mm_opencl_intelfpga_backend_crc_ui.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend_crc_ui.aocx $(FPGA_DEF) -D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CRC

#FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_crc_db.aocx mm_opencl_intelfpga_backend_crc_fl.aocx mm_opencl_intelfpga_backend_crc_ul.aocx mm_opencl_intelfpga_backend_crc_in.aocx mm_opencl_intelfpga_backend_crc_ui.aocx
FPGA_DEPS := $(FPGA_DEPS) mm_opencl_intelfpga_backend_crc_ui.aocx
FPGA_DEF := -D OPENCL_SINGLE_KERNEL_PROGS $(FPGA_DEF)
else
FPGA_DEPS = mm_opencl_intelfpga_backend.aocx
endif
endif
ifeq ($(OPENCL_SINGLE_KERNEL_PROGS),TRUE)
 CL_DEFS := $(CL_DEFS) -D OPENCL_SINGLE_KERNEL_PROGS
endif
ifeq ($(USE_TIMERS),TRUE)
 CL_DEFS := $(CL_DEFS) -D WITH_TIMERS
endif

libmm_opencl_backend.so:
	g++ mm_opencl_backend.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmm_opencl_backend.so $(CL_DEFS) -shared -Wl,-soname,libmm_opencl_backend.so -DMETAMORPH_OCL_KERNEL_PATH=\"$(shell pwd):$(MM_LIB):.\"
#deprecated
#libmm_opencl_intelfpga_backend.so: $(FPGA_DEPS)
#	g++ mm_opencl_intelfpga_backend.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) $(FPGA_DEF) $(AOCL_COMPILE_CONFIG) $(FPGA_LIB) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmm_opencl_intelfpga_backend.so -D __OPENCLCC__ -shared -Wl,-soname,libmm_opencl_intelfpga_backend.so


libmm_opencl_intelfpga_backend.so: $(FPGA_DEPS)
	g++ mm_opencl_backend.cpp $(OPT_LVL) $(L_FLAGS) $(INCLUDES) $(FPGA_DEF) $(AOCL_COMPILE_CONFIG) $(FPGA_LIB) -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL -o $(MM_LIB)/libmm_opencl_intelfpga_backend.so -D __OPENCLCC__ -shared -Wl,-soname,libmm_opencl_intelfpga_backend.so -DMETAMORPH_OCL_KERNEL_PATH=\"$(shell pwd):$(MM_LIB):.\"

mm_opencl_intelfpga_backend.aocx: mm_opencl_backend.cl
	aoc $(AOC_DEF) mm_opencl_backend.cl -o $(MM_LIB)/mm_opencl_intelfpga_backend.aocx $(FPGA_DEF)
