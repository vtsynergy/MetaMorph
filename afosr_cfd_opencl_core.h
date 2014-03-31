#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


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

		cl_kernel kernel_reduction3;
		cl_kernel kernel_dotProd;
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

	cl_int opencl_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], double * data1, double * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val, int async, cl_event * event); 
	cl_int opencl_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], double * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val, int async, cl_event * event); 


#ifdef __OPENCLCC__
}
#endif
