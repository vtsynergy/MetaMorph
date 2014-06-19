#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef AFOSR_CFD_OPENCL_CORE_H
#define AFOSR_CFD_OPENCL_CORE_H

#ifndef AFOSR_CFD_H
	#include "afosr_cfd.h"
#endif

//Not sure if these C compatibility stubs will actually be needed
#ifdef __OPENCLCC__
extern "C" {
#endif

	//This is the exposed object for managing multiple OpenCL devices with the
	// library's built-in stack.
	//These frame objects are allocated and freed by the Init and Destroy calls, respectively
	//All Stack operations (push, top and pop) make a copy of the frame's contents, therefore,
	// the frame itself is thread-private, and even if tampered with, cannot affect the thread
	// safety of the shared stack object.
	//For basic CPU/GPU/MIC devices, the provided initialization calls should be sufficient and
	// frame objects should not need to be directly modified; however, for FPGA devices which must
	// use clBuildProgramFromBinary, the user will need to implement analogues to
	// accelOpenCLInitStackFrame and accelOpenCLDestroyStackFrame, which appropriately create and
	// release all necessary frame members. If implemented correctly, the built-in hazard-aware
	// stack operations shouldn't need any changes.
	typedef struct  accelOpenCLStackFrame
	{

		cl_platform_id platform;
		cl_device_id device;
		cl_context context;
		cl_command_queue queue;

		cl_program program_opencl_core;

		cl_kernel kernel_reduce_db;
		cl_kernel kernel_reduce_fl;
		cl_kernel kernel_reduce_ul;
		cl_kernel kernel_reduce_in;
		cl_kernel kernel_reduce_ui;
		cl_kernel kernel_dotProd_db;
		cl_kernel kernel_dotProd_fl;
		cl_kernel kernel_dotProd_ul;
		cl_kernel kernel_dotProd_in;
		cl_kernel kernel_dotProd_ui;
		cl_kernel kernel_transpose_2d_face_db;
		cl_kernel kernel_transpose_2d_face_fl;
		cl_kernel kernel_transpose_2d_face_ul;
		cl_kernel kernel_transpose_2d_face_in;
		cl_kernel kernel_transpose_2d_face_ui;
		cl_kernel kernel_pack_2d_face_db;
		cl_kernel kernel_pack_2d_face_fl;
		cl_kernel kernel_pack_2d_face_ul;
		cl_kernel kernel_pack_2d_face_in;
		cl_kernel kernel_pack_2d_face_ui;
		cl_kernel kernel_unpack_2d_face_db;
		cl_kernel kernel_unpack_2d_face_fl;
		cl_kernel kernel_unpack_2d_face_ul;
		cl_kernel kernel_unpack_2d_face_in;
		cl_kernel kernel_unpack_2d_face_ui;

		cl_mem constant_face_size;
		cl_mem constant_face_stride;
		cl_mem constant_face_child_size;

	} accelOpenCLStackFrame;
	//TODO these shouldn't need to be exposed to the user, unless there's a CUDA call we need to emulate
	void accelOpenCLPushStackFrame(accelOpenCLStackFrame * frame);

	accelOpenCLStackFrame * accelOpenCLTopStackFrame();

	accelOpenCLStackFrame * accelOpenCLPopStackFrame();

	//start everything for a frame
	cl_int accelOpenCLInitStackFrame(accelOpenCLStackFrame ** frame, cl_int device);

	//stop everything for a frame
	cl_int accelOpenCLDestroyStackFrame(accelOpenCLStackFrame * frame);

	//support initialization of a default frame as well as environment variable
	// -based control, via $TARGET_DEVICE="Some Device Name"
	cl_int accelOpenCLInitStackFrameDefault(accelOpenCLStackFrame ** frame);

	size_t accelOpenCLLoadProgramSource(const char *filename, const char **progSrc);

	cl_int opencl_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], void * data1, void * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, accel_type_id type, int async, cl_event * event); 
	cl_int opencl_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], void * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, accel_type_id type, int async, cl_event * event); 
	cl_int opencl_transpose_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *indata, void *outdata, size_t (* dim_xy)[3], accel_type_id type, int async, cl_event * event);
	cl_int opencl_pack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, accel_2d_face_indexed *face, int *remain_dim, accel_type_id type, int async, cl_event * event_k1, cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);
	cl_int opencl_unpack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, accel_2d_face_indexed *face, int *remain_dim, accel_type_id type, int async, cl_event * event_k1, cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);

#ifdef __OPENCLCC__
}
#endif

#endif //AFOSR_CFD_OPENCL_CORE_H
