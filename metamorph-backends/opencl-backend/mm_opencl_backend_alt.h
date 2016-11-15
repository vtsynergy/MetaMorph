#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/** OpenCL Back-End: FPGA customization **/
#ifndef METAMORPH_OPENCL_BACKEND_H
#define METAMORPH_OPENCL_BACKEND_H

#ifndef METAMORPH_H
#include "metamorph.h"
#endif

//If the user doesn't override default threadblock size..
#ifndef METAMORPH_OCL_DEFAULT_BLOCK
#define METAMORPH_OCL_DEFAULT_BLOCK {16, 8, 1}
#endif

#ifndef __FPGA__
#define FPGA_DOUBLE
#define FPGA_FLOAT
#define FPGA_UNSIGNED_LONG
#define FPGA_INTEGER
#define FPGA_UNSIGNED_INTEGER
#define KERNEL_REDUCE
#define KERNEL_DOT_PROD
#define KERNEL_TRANSPOSE
#define KERNEL_PACK
#define KERNEL_UPACK
#define KERNEL_STENCIL
#endif 

#if (!defined(FPGA_DOUBLE) && !defined(FPGA_FLOAT) && !defined(FPGA_UNSIGNED_LONG) && !defined(FPGA_INTEGER) && !defined(FPGA_UNSIGNED_INTEGER))
#error Macro is Undefined,Please define one of FPGA_DOUBLE, FPGA_FLOAT, FPGA_UNSIGNED_LONG, FPGA_INTEGER, FPGA_UNSIGNED_INTEGER
#endif 

#if (!defined(KERNEL_REDUCE) && !defined(KERNEL_DOT_PROD) && !defined(KERNEL_TRANSPOSE) && !defined(KERNEL_PACK) && !defined(KERNEL_UPACK) && !defined(KERNEL_STENCIL))
#error Macro is undefined. Define at least one of the kernel.
#endif 

//Not sure if these C compatibility stubs will actually be needed
#ifdef __OPENCLCC__
extern "C" {
#endif

//This is the exposed object for managing multiple OpenCL devices with the
// library's built-in stack.
//These frame objects are allocated and freed by the Init and Destroy calls, respectively.
//All Stack operations (push, top and pop) make a copy of the frame's contents, therefore,
// the frame itself is thread-private, and even if tampered with, cannot affect the thread
// safety of the shared stack object.
//For basic CPU/GPU/MIC devices, the provided initialization calls should be sufficient and
// frame objects should not need to be directly modified; however, for FPGA devices which must
// use clBuildProgramFromBinary, the user will need to implement analogues to
// metaOpenCLInitStackFrame and metaOpenCLDestroyStackFrame, which appropriately create and
// release all necessary frame members. If implemented correctly, the built-in hazard-aware
// stack operations shouldn't need any changes.
typedef struct metaOpenCLStackFrame {

	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;

	cl_program program_opencl_core;

#ifdef FPGA_DOUBLE
#ifdef KERNEL_REDUCE
	cl_kernel kernel_reduce_db;
#endif // KERNEL_REDUCE
#ifdef KERNEL_DOT_PROD
	cl_kernel kernel_dotProd_db;
#endif // KERNEL_DOT_PROD
#ifdef KERNEL_TRANSPOSE
	cl_kernel kernel_transpose_2d_face_db;
#endif // KERNEL_TRANSPOSE
#ifdef KERNEL_PACK
	cl_kernel kernel_pack_2d_face_db;
#endif // KERNEL_PACK
#ifdef KERNEL_UPACK
	cl_kernel kernel_unpack_2d_face_db;
#endif // KERNEL_UPACK
#ifdef KERNEL_STENCIL
	cl_kernel kernel_stencil_3d7p_db;
#endif // KERNL_STENCIL
#endif // FPGA_DOUBLE
#ifdef  FPGA_FLOAT
#ifdef KERNEL_REDUCE
	cl_kernel kernel_reduce_fl;
#endif // KERNEL_REDUCE
#ifdef KERNEL_DOT_PROD
	cl_kernel kernel_dotProd_fl;
#endif // KERNEL_DOT_PROD
#ifdef KERNEL_TRANSPOSE
	cl_kernel kernel_transpose_2d_face_fl;
#endif // KERNEL_TRANSPOSE
#ifdef KERNEL_PACK
	cl_kernel kernel_pack_2d_face_fl;
#endif // KERNEL_PACK
#ifdef KERNEL_UPACK
	cl_kernel kernel_unpack_2d_face_fl;
#endif // KERNEL_UPACK
#ifdef KERNEL_STENCIL
	cl_kernel kernel_stencil_3d7p_fl;
#endif // KERNL_STENCIL
#endif // FPGA_FLOAT
#ifdef FPGA_UNSIGNED_LONG
#ifdef KERNEL_REDUCE
	cl_kernel kernel_reduce_ul;
#endif // KERNEL_REDUCE
#ifdef KERNEL_DOT_PROD
	cl_kernel kernel_dotProd_ul;
#endif // KERNEL_DOT_PROD
#ifdef KERNEL_TRANSPOSE
	cl_kernel kernel_transpose_2d_face_ul;
#endif // KERNEL_TRANSPOSE
#ifdef KERNEL_PACK
	cl_kernel kernel_pack_2d_face_ul;
#endif // KERNEL_PACK
#ifdef KERNEL_UPACK
	cl_kernel kernel_unpack_2d_face_ul;
#endif // KERNEL_UPACK
#ifdef KERNEL_STENCIL
	cl_kernel kernel_stencil_3d7p_ul;
#endif // KERNL_STENCIL
#endif // FPGA_UNSIGNED_LONG
#ifdef FPGA_INTEGER
#ifdef KERNEL_REDUCE
	cl_kernel kernel_reduce_in;
#endif // KERNEL_REDUCE
#ifdef KERNEL_DOT_PROD
	cl_kernel kernel_dotProd_in;
#endif // KERNEL_DOT_PROD
#ifdef KERNEL_TRANSPOSE
	cl_kernel kernel_transpose_2d_face_in;
#endif // KERNEL_TRANSPOSE
#ifdef KERNEL_PACK
	cl_kernel kernel_pack_2d_face_in;
#endif // KERNEL_PACK
#ifdef KERNEL_UPACK
	cl_kernel kernel_unpack_2d_face_in;
#endif // KERNEL_UPACK
#ifdef KERNEL_STENCIL
	cl_kernel kernel_stencil_3d7p_in;
#endif // KERNL_STENCIL
#endif // FPGA_INTEGER
#ifdef FPGA_UNSIGNED_INTEGER
#ifdef KERNEL_REDUCE
	cl_kernel kernel_reduce_ui;
#endif // KERNEL_REDUCE
#ifdef KERNEL_DOT_PROD
	cl_kernel kernel_dotProd_ui;
#endif // KERNEL_DOT_PROD
#ifdef KERNEL_TRANSPOSE
	cl_kernel kernel_transpose_2d_face_ui;
#endif // KERNEL_TRANSPOSE
#ifdef KERNEL_PACK
	cl_kernel kernel_pack_2d_face_ui;
#endif // KERNEL_PACK
#ifdef KERNEL_UPACK
	cl_kernel kernel_unpack_2d_face_ui;
#endif // KERNEL_UPACK
#ifdef KERNEL_STENCIL
	cl_kernel kernel_stencil_3d7p_ui;
#endif // KERNL_STENCIL
#endif // FPGA_UNSIGNED_INTEGER 

	cl_mem constant_face_size;
	cl_mem constant_face_stride;
	cl_mem constant_face_child_size;
	cl_mem red_loc;

} metaOpenCLStackFrame;
//TODO these shouldn't need to be exposed to the user, unless there's a CUDA call we need to emulate
void metaOpenCLPushStackFrame(metaOpenCLStackFrame * frame);

metaOpenCLStackFrame * metaOpenCLTopStackFrame();

metaOpenCLStackFrame * metaOpenCLPopStackFrame();

//start everything for a frame
cl_int metaOpenCLInitStackFrame(metaOpenCLStackFrame ** frame, cl_int device);

//stop everything for a frame
cl_int metaOpenCLDestroyStackFrame(metaOpenCLStackFrame * frame);

//support initialization of a default frame as well as environment variable
// -based control, via $TARGET_DEVICE="Some Device Name"
cl_int metaOpenCLInitStackFrameDefault(metaOpenCLStackFrame ** frame);

size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc);

cl_int opencl_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data1, void * data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], void * reduced_val,
		meta_type_id type, int async, cl_event * event);
cl_int opencl_reduce(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data, size_t (*array_size)[3], size_t (*arr_start)[3],
		size_t (*arr_end)[3], void * reduced_val, meta_type_id type, int async,
		cl_event * event);
cl_int opencl_transpose_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *indata, void *outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3], meta_type_id type, int async,
		cl_event * event);
cl_int opencl_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);
cl_int opencl_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);
cl_int opencl_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * indata, void * outdata, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], meta_type_id type,
		int async, cl_event * event);

#ifdef __OPENCLCC__
}
#endif

#endif //METAMORPH_OPENCL_BACKEND_H
