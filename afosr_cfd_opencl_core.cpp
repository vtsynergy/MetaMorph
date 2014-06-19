#include "afosr_cfd_opencl_core.h"

//Warning, none of these variables are threadsafe, so only one thread should ever
// perform the one-time scan!!
cl_uint num_platforms, num_devices;
cl_platform_id * platforms = NULL;
cl_device_id * devices = NULL;


typedef struct accelOpenCLStackNode
{
	accelOpenCLStackFrame frame;
	struct accelOpenCLStackNode * next;
} accelOpenCLStackNode;

accelOpenCLStackNode * CLStack = NULL;


const char *accelCLProgSrc;
size_t accelCLProgLen;


//Returns the first id in the device array of the desired device, or -1 if no such device is present
int accelOpenCLGetDeviceID(char * desired, cl_device_id * devices, int numDevices) {
	char buff[128];
	int i;
	for (i = 0; i < numDevices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, (void *)buff, NULL);
		//printf("%s\n", buff);
		if (strcmp(desired, buff) == 0) return i;
	}
	return -1;
}

//Basic one-time scan of all platforms for all devices
//This is not threadsafe, and I'm not sure we'll ever make it safe
//If we need to though, it would likely be sufficient to CAS the num_platforms int or platforms pointer
// first to a hazard flag (like -1) to claim it, then to the actual pointer.
void accelOpenCLQueryDevices() {
	int i;
	num_platforms = 0, num_devices = 0;
	cl_uint temp_uint, temp_uint2;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) printf("Failed to query platform count!\n");
	printf("Number of OpenCL Platforms: %d\n", num_platforms);

	platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);

	if (clGetPlatformIDs(num_platforms, &platforms[0], NULL) != CL_SUCCESS) printf("Failed to get platform IDs\n");

	for (i = 0; i < num_platforms; i++) {
		temp_uint = 0;
		if(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &temp_uint) != CL_SUCCESS) printf("Failed to query device count on platform %d!\n", i);
		num_devices += temp_uint;
	}
	printf("Number of Devices: %d\n", num_devices);

	devices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
	temp_uint = 0;
	for ( i = 0; i < num_platforms; i++) {
		if(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, &devices[temp_uint], &temp_uint2) != CL_SUCCESS) printf ("Failed to query device IDs on platform %d!\n", i);
		temp_uint += temp_uint2;
		temp_uint2 = 0;
	}

	//TODO figure out somewhere else to put this, like a terminating callback or something...
	//free(devices);
	//free(platforms);
}

size_t accelOpenCLLoadProgramSource(const char *filename, const char **progSrc) {
	FILE *f = fopen(filename, "r");
	fseek(f, 0, SEEK_END);
	size_t len = (size_t) ftell(f);
	*progSrc = (const char *) malloc(sizeof(char)*len);
	rewind(f);
	fread((void *) *progSrc, len, 1, f);
	fclose(f);
	return len;
}



//This should not be exposed to the user, just the top and pop functions built on top of it
//for thread-safety, returns a new accelOpenCLStackNode, which is a direct copy
// of the top at some point in time.
// this way, the user can still use top, without having to manage hazard pointers themselves
//ASSUME HAZARD POINTERS ARE ALREADY SET FOR t BY THE CALLING METHOD
void copyStackNodeToFrame(accelOpenCLStackNode * t, accelOpenCLStackFrame ** frame) {

	//From here out, we have hazards
	//Copy all the parameters - REALLY HAZARDOUS

	//Top-level context info
	(*frame)->platform = t->frame.platform;
	(*frame)->device = t->frame.device;
	(*frame)->context = t->frame.context;
	(*frame)->queue = t->frame.queue;
	(*frame)->program_opencl_core = t->frame.program_opencl_core;

	//Kernels
	(*frame)->kernel_reduce_db = t->frame.kernel_reduce_db;
	(*frame)->kernel_reduce_fl = t->frame.kernel_reduce_fl;
	(*frame)->kernel_reduce_ul = t->frame.kernel_reduce_ul;
	(*frame)->kernel_reduce_in = t->frame.kernel_reduce_in;
	(*frame)->kernel_reduce_ui = t->frame.kernel_reduce_ui;
	(*frame)->kernel_dotProd_db = t->frame.kernel_dotProd_db;
	(*frame)->kernel_dotProd_fl = t->frame.kernel_dotProd_fl;
	(*frame)->kernel_dotProd_ul = t->frame.kernel_dotProd_ul;
	(*frame)->kernel_dotProd_in = t->frame.kernel_dotProd_in;
	(*frame)->kernel_dotProd_ui = t->frame.kernel_dotProd_ui;
	(*frame)->kernel_transpose_2d_face_db = t->frame.kernel_transpose_2d_face_db;
	(*frame)->kernel_transpose_2d_face_fl = t->frame.kernel_transpose_2d_face_fl;
	(*frame)->kernel_transpose_2d_face_ul = t->frame.kernel_transpose_2d_face_ul;
	(*frame)->kernel_transpose_2d_face_in = t->frame.kernel_transpose_2d_face_in;
	(*frame)->kernel_transpose_2d_face_ui = t->frame.kernel_transpose_2d_face_ui;
	(*frame)->kernel_pack_2d_face_db = t->frame.kernel_pack_2d_face_db;
	(*frame)->kernel_pack_2d_face_fl = t->frame.kernel_pack_2d_face_fl;
	(*frame)->kernel_pack_2d_face_ul = t->frame.kernel_pack_2d_face_ul;
	(*frame)->kernel_pack_2d_face_in = t->frame.kernel_pack_2d_face_in;
	(*frame)->kernel_pack_2d_face_ui = t->frame.kernel_pack_2d_face_ui;
	(*frame)->kernel_unpack_2d_face_db = t->frame.kernel_unpack_2d_face_db;
	(*frame)->kernel_unpack_2d_face_fl = t->frame.kernel_unpack_2d_face_fl;
	(*frame)->kernel_unpack_2d_face_ul = t->frame.kernel_unpack_2d_face_ul;
	(*frame)->kernel_unpack_2d_face_in = t->frame.kernel_unpack_2d_face_in;
	(*frame)->kernel_unpack_2d_face_ui = t->frame.kernel_unpack_2d_face_ui;
	
	//Internal buffers
	(*frame)->constant_face_size = t->frame.constant_face_size;
	(*frame)->constant_face_stride = t->frame.constant_face_stride;
	(*frame)->constant_face_child_size = t->frame.constant_face_child_size;

	//This should be the end of the hazards;
}

//ASSUME HAZARD POINTERS ARE ALREADY SET FOR node BY THE CALLING METHOD
void copyStackFrameToNode(accelOpenCLStackFrame * f, accelOpenCLStackNode ** node) {

	//Top-level context info
	(*node)->frame.platform = f->platform;
	(*node)->frame.device = f->device;
	(*node)->frame.context = f->context;
	(*node)->frame.queue = f->queue;
	(*node)->frame.program_opencl_core = f->program_opencl_core;

	//Kernels
	(*node)->frame.kernel_reduce_db = f->kernel_reduce_db;
	(*node)->frame.kernel_reduce_fl = f->kernel_reduce_fl;
	(*node)->frame.kernel_reduce_ul = f->kernel_reduce_ul;
	(*node)->frame.kernel_reduce_in = f->kernel_reduce_in;
	(*node)->frame.kernel_reduce_ui = f->kernel_reduce_ui;
	(*node)->frame.kernel_dotProd_db = f->kernel_dotProd_db;
	(*node)->frame.kernel_dotProd_fl = f->kernel_dotProd_fl;
	(*node)->frame.kernel_dotProd_ul = f->kernel_dotProd_ul;
	(*node)->frame.kernel_dotProd_in = f->kernel_dotProd_in;
	(*node)->frame.kernel_dotProd_ui = f->kernel_dotProd_ui;
	(*node)->frame.kernel_transpose_2d_face_db = f->kernel_transpose_2d_face_db;
	(*node)->frame.kernel_transpose_2d_face_fl = f->kernel_transpose_2d_face_fl;
	(*node)->frame.kernel_transpose_2d_face_ul = f->kernel_transpose_2d_face_ul;
	(*node)->frame.kernel_transpose_2d_face_in = f->kernel_transpose_2d_face_in;
	(*node)->frame.kernel_transpose_2d_face_ui = f->kernel_transpose_2d_face_ui;
	(*node)->frame.kernel_pack_2d_face_db = f->kernel_pack_2d_face_db;
	(*node)->frame.kernel_pack_2d_face_fl = f->kernel_pack_2d_face_fl;
	(*node)->frame.kernel_pack_2d_face_ul = f->kernel_pack_2d_face_ul;
	(*node)->frame.kernel_pack_2d_face_in = f->kernel_pack_2d_face_in;
	(*node)->frame.kernel_pack_2d_face_ui = f->kernel_pack_2d_face_ui;
	(*node)->frame.kernel_unpack_2d_face_db = f->kernel_unpack_2d_face_db;
	(*node)->frame.kernel_unpack_2d_face_fl = f->kernel_unpack_2d_face_fl;
	(*node)->frame.kernel_unpack_2d_face_ul = f->kernel_unpack_2d_face_ul;
	(*node)->frame.kernel_unpack_2d_face_in = f->kernel_unpack_2d_face_in;
	(*node)->frame.kernel_unpack_2d_face_ui = f->kernel_unpack_2d_face_ui;

	//Internal Buffers
	(*node)->frame.constant_face_size = f->constant_face_size;
	(*node)->frame.constant_face_stride = f->constant_face_stride;
	(*node)->frame.constant_face_child_size = f->constant_face_child_size;
}

//For this push to be both exposed and safe from hazards, it must make a copy of the frame
// such that the user never has access to the pointer to the frame that's actually on the shared stack
// object
//This CAS-based push should be threadsafe
//WARNING assumes all parameters of accelOpenCLStackNode are set, except "next"
void accelOpenCLPushStackFrame(accelOpenCLStackFrame * frame) {
	//copy the frame, this is still the thread-private "allocated" state
	accelOpenCLStackNode * newNode = (accelOpenCLStackNode *) malloc (sizeof(accelOpenCLStackNode));
	copyStackFrameToNode(frame, &newNode);


	//grab the old node
	accelOpenCLStackNode * old = CLStack;
	//I think this is where hazards start..
	newNode->next = old;
	CLStack = newNode;

}

accelOpenCLStackFrame * accelOpenCLTopStackFrame() { 
	accelOpenCLStackFrame * frame = (accelOpenCLStackFrame * ) malloc(sizeof(accelOpenCLStackFrame));
	accelOpenCLStackNode * t = CLStack; //Hazards start
	//so set a hazard pointer
	//then copy the node
	copyStackNodeToFrame(t, &frame);
	//release the hazard pointer
	//so hazards are over, return the copy and exit
	return(frame);
}

accelOpenCLStackFrame * accelOpenCLPopStackFrame() {
	accelOpenCLStackFrame * frame = (accelOpenCLStackFrame *) malloc(sizeof(accelOpenCLStackFrame));
	accelOpenCLStackNode * t = CLStack; //Hazards start
	//so set a hazard pointer
	//then copy the node
	copyStackNodeToFrame(t, &frame);
	//and pull the node off the stack
	CLStack = t->next;
	//Do something with the memory
	//free(t);
	//release the hazard pointer
	//so hazards are over, return the copy and exit
	return(frame);
}



cl_int accelOpenCLInitStackFrame(accelOpenCLStackFrame ** frame, cl_int device) {
	cl_int ret = CL_SUCCESS;
	//First, make sure we've run the one-time query to initialize the device array
	//TODO, fix synchronization on the one-time device query.
	if (platforms == NULL || devices == NULL || (((long) platforms) == -1)) {
		//try to perform the scan, else wait while somebody else finishes it.
		accelOpenCLQueryDevices();
	}

	//Hack to allow choose device to set mode to OpenCL, but still pick a default
	if (device == -1) return accelOpenCLInitStackFrameDefault(frame);


	*frame = (accelOpenCLStackFrame *) malloc(sizeof(accelOpenCLStackFrame));
	//TODO use the device array to do reverse lookup to figure out which platform to attach to the frame
	//TODO retrofit to take an integer parameter indexing into the device array
	//TODO ensure the device array has been initialized by somebody (even if I have to do it) before executing this blook
	//TODO implement an intelligent catch for if the device number is out of range

	//copy the chosen device from the array to the new frame
	(*frame)->device = devices[device];
	//reverse lookup the device's platform and add it to the frame
	clGetDeviceInfo((*frame)->device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &((*frame)->platform), NULL);
	//create the context and add it to the frame
	(*frame)->context = clCreateContext(NULL, 1, &((*frame)->device), NULL, NULL, NULL);
	(*frame)->queue = clCreateCommandQueue((*frame)->context, (*frame)->device, CL_QUEUE_PROFILING_ENABLE, NULL);
	if (accelCLProgLen == 0) {
		accelCLProgLen = accelOpenCLLoadProgramSource("afosr_cfd_opencl_core.cl", &accelCLProgSrc);
	}
	(*frame)->program_opencl_core = clCreateProgramWithSource((*frame)->context, 1, &accelCLProgSrc, &accelCLProgLen, NULL);
	// Add this debug string if needed: -g -s\"./afosr_cfd_opencl_core.cl\"
	char * args = NULL;
	if (getenv("AFOSR_MODE") != NULL) {
		if (strcmp(getenv("AFOSR_MODE"), "OpenCL") == 0) {
			size_t needed = snprintf(NULL, 0, "-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d)", TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS);
			args = (char *)malloc(needed);
			snprintf(args, needed, "-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d)", TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS);
			ret |= clBuildProgram((*frame)->program_opencl_core, 1, &((*frame)->device), args, NULL, NULL);
		}
		else if (strcmp(getenv("AFOSR_MODE"), "OpenCL_DEBUG") == 0) {
			size_t needed = snprintf(NULL, 0, "-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) -g -cl-opt-disable", TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS);
			args = (char *)malloc(needed);
			snprintf(args, needed, "-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) -g -cl-opt-disable", TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS);
		}
	}
	ret |= clBuildProgram((*frame)->program_opencl_core, 1, &((*frame)->device), args, NULL, NULL);
	//Let us know if there's any errors in the build process
	if (ret != CL_SUCCESS) {fprintf(stderr, "Error in clBuildProgram: %d!\n", ret); ret = CL_SUCCESS;}
	//Stub to get build log
	size_t logsize = 0;
	clGetProgramBuildInfo((*frame)->program_opencl_core, (*frame)->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
	char * log = (char *) malloc (sizeof(char) *(logsize+1));
	clGetProgramBuildInfo((*frame)->program_opencl_core, (*frame)->device, CL_PROGRAM_BUILD_LOG, logsize, log, NULL);
	fprintf(stderr, "CL_PROGRAM_BUILD_LOG:\n%s", log);
	free(log);
	(*frame)->kernel_reduce_db = clCreateKernel((*frame)->program_opencl_core, "kernel_reduce_db", NULL);
	(*frame)->kernel_reduce_fl = clCreateKernel((*frame)->program_opencl_core, "kernel_reduce_fl", NULL);
	(*frame)->kernel_reduce_ul = clCreateKernel((*frame)->program_opencl_core, "kernel_reduce_ul", NULL);
	(*frame)->kernel_reduce_in = clCreateKernel((*frame)->program_opencl_core, "kernel_reduce_in", NULL);
	(*frame)->kernel_reduce_ui = clCreateKernel((*frame)->program_opencl_core, "kernel_reduce_ui", NULL);
	(*frame)->kernel_dotProd_db = clCreateKernel((*frame)->program_opencl_core, "kernel_dotProd_db", NULL);
	(*frame)->kernel_dotProd_fl = clCreateKernel((*frame)->program_opencl_core, "kernel_dotProd_fl", NULL);
	(*frame)->kernel_dotProd_ul = clCreateKernel((*frame)->program_opencl_core, "kernel_dotProd_ul", NULL);
	(*frame)->kernel_dotProd_in = clCreateKernel((*frame)->program_opencl_core, "kernel_dotProd_in", NULL);
	(*frame)->kernel_dotProd_ui = clCreateKernel((*frame)->program_opencl_core, "kernel_dotProd_ui", NULL);
	(*frame)->kernel_transpose_2d_face_db = clCreateKernel((*frame)->program_opencl_core, "kernel_transpose_2d_db", NULL);
	(*frame)->kernel_transpose_2d_face_fl = clCreateKernel((*frame)->program_opencl_core, "kernel_transpose_2d_fl", NULL);
	(*frame)->kernel_transpose_2d_face_ul = clCreateKernel((*frame)->program_opencl_core, "kernel_transpose_2d_ul", NULL);
	(*frame)->kernel_transpose_2d_face_in = clCreateKernel((*frame)->program_opencl_core, "kernel_transpose_2d_in", NULL);
	(*frame)->kernel_transpose_2d_face_ui = clCreateKernel((*frame)->program_opencl_core, "kernel_transpose_2d_ui", NULL);
	(*frame)->kernel_pack_2d_face_db = clCreateKernel((*frame)->program_opencl_core, "kernel_pack_db", NULL);
	(*frame)->kernel_pack_2d_face_fl = clCreateKernel((*frame)->program_opencl_core, "kernel_pack_fl", NULL);
	(*frame)->kernel_pack_2d_face_ul = clCreateKernel((*frame)->program_opencl_core, "kernel_pack_ul", NULL);
	(*frame)->kernel_pack_2d_face_in = clCreateKernel((*frame)->program_opencl_core, "kernel_pack_in", NULL);
	(*frame)->kernel_pack_2d_face_ui = clCreateKernel((*frame)->program_opencl_core, "kernel_pack_ui", NULL);
	(*frame)->kernel_unpack_2d_face_db = clCreateKernel((*frame)->program_opencl_core, "kernel_unpack_db", NULL);
	(*frame)->kernel_unpack_2d_face_fl = clCreateKernel((*frame)->program_opencl_core, "kernel_unpack_fl", NULL);
	(*frame)->kernel_unpack_2d_face_ul = clCreateKernel((*frame)->program_opencl_core, "kernel_unpack_ul", NULL);
	(*frame)->kernel_unpack_2d_face_in = clCreateKernel((*frame)->program_opencl_core, "kernel_unpack_in", NULL);
	(*frame)->kernel_unpack_2d_face_ui = clCreateKernel((*frame)->program_opencl_core, "kernel_unpack_ui", NULL);

	//Allocate any internal buffers necessary for kernel functions
	(*frame)->constant_face_size = clCreateBuffer((*frame)->context, CL_MEM_READ_ONLY, sizeof(cl_int)*AFOSR_FACE_MAX_DEPTH, NULL, NULL);
	(*frame)->constant_face_stride = clCreateBuffer((*frame)->context, CL_MEM_READ_ONLY, sizeof(cl_int)*AFOSR_FACE_MAX_DEPTH, NULL, NULL);
	(*frame)->constant_face_child_size = clCreateBuffer((*frame)->context, CL_MEM_READ_ONLY, sizeof(cl_int)*AFOSR_FACE_MAX_DEPTH, NULL, NULL);
}

//calls all the necessary CLRelease* calls for frame members
//DOES NOT:
//	pop any stack nodes
//	free any stack nodes
//	free program source
//	implement any thread safety, frames should always be thread private, only the stack should be shared, and all franes must be copied to or from stack nodes using the hazard-aware copy methods.
//	 (more specifically, copying a frame to a node doesn't need to be hazard-aware, as the node cannot be shared unless copied inside the hazard-aware accelOpenCLPushStackFrame. Pop, Top, and copyStackNodeToFrame are all hazard aware and provide a thread-private copy back to the caller.)
cl_int accelOpenCLDestroyStackFrame(accelOpenCLStackFrame * frame) {

	//Release Kernels
	clReleaseKernel(frame->kernel_reduce_db);
	clReleaseKernel(frame->kernel_reduce_fl);
	clReleaseKernel(frame->kernel_reduce_ul);
	clReleaseKernel(frame->kernel_reduce_in);
	clReleaseKernel(frame->kernel_reduce_ui);
	clReleaseKernel(frame->kernel_dotProd_db);
	clReleaseKernel(frame->kernel_dotProd_fl);
	clReleaseKernel(frame->kernel_dotProd_ul);
	clReleaseKernel(frame->kernel_dotProd_in);
	clReleaseKernel(frame->kernel_dotProd_ui);
	clReleaseKernel(frame->kernel_transpose_2d_face_db);
	clReleaseKernel(frame->kernel_transpose_2d_face_fl);
	clReleaseKernel(frame->kernel_transpose_2d_face_ul);
	clReleaseKernel(frame->kernel_transpose_2d_face_in);
	clReleaseKernel(frame->kernel_transpose_2d_face_ui);
	clReleaseKernel(frame->kernel_pack_2d_face_db);
	clReleaseKernel(frame->kernel_pack_2d_face_fl);
	clReleaseKernel(frame->kernel_pack_2d_face_ul);
	clReleaseKernel(frame->kernel_pack_2d_face_in);
	clReleaseKernel(frame->kernel_pack_2d_face_ui);
	clReleaseKernel(frame->kernel_unpack_2d_face_db);
	clReleaseKernel(frame->kernel_unpack_2d_face_fl);
	clReleaseKernel(frame->kernel_unpack_2d_face_ul);
	clReleaseKernel(frame->kernel_unpack_2d_face_in);
	clReleaseKernel(frame->kernel_unpack_2d_face_ui);

	//Release Internal Buffers
	clReleaseMemObject(frame->constant_face_size);
	clReleaseMemObject(frame->constant_face_stride);
	clReleaseMemObject(frame->constant_face_child_size);

	//Release remaining context info
	clReleaseProgram(frame->program_opencl_core);
	clReleaseCommandQueue(frame->queue);
	clReleaseContext(frame->context);

	//TODO since this only destroys a frame, we must release the global program source elsewhere
}

//This is a fallback catchall to ensure some context is initialized
// iff the user calls an accelerator function in OpenCL mode before
// calling choose_accel in OpenCL mode.
//It will pick some valid OpenCL device, and emit a warning to stderr
// that all OpenCL calls until choose_accel will refer to this device
//It also implements environment-variable-controlled device selection
// via the "TARGET_DEVICE" string environemnt variable, which must match
// EXACTLY the device name reported to the OpenCL runtime.
//TODO implement a reasonable passthrough for any errors which OpenCL may throw.
cl_int accelOpenCLInitStackFrameDefault(accelOpenCLStackFrame ** frame) {
	cl_int ret = CL_SUCCESS;
	//First, make sure we've run the one-time query to initialize the device array
	if (platforms == NULL || devices == NULL || (((long) platforms) == -1)) {
		//try to perform the scan, else wait while somebody else finishes it.
		accelOpenCLQueryDevices();
	}

	//Simply print the names of all devices, to assist later environment-variable device selection
	fprintf(stderr, "WARNING: Automatic OpenCL device selection used!\n");
	fprintf(stderr, "\tThe following devices are identified in the system:\n");
	char buff[128];
	int i;
	for (i = 0; i < num_devices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, (void *)&buff[0], NULL);
		fprintf(stderr, "Device [%d]: \"%s\"\n", i, buff);
	}

	//This is how you pick a specific device using an environment variable

	int gpuID =-1;

	if (getenv("TARGET_DEVICE") != NULL) {
		gpuID = accelOpenCLGetDeviceID(getenv("TARGET_DEVICE"), &devices[0], num_devices);
		if (gpuID < 0) fprintf(stderr, "Device \"%s\" not found.\nDefaulting to first device found.\n", getenv("TARGET_DEVICE"));
	} else {
		fprintf(stderr, "Environment variable TARGET_DEVICE not set.\nDefaulting to first device found.\n");
	}

	gpuID = gpuID < 0 ? 0 : gpuID; //Ternary check to make sure gpuID is valid, if it's less than zero, default to zero, otherwise keep

	clGetDeviceInfo(devices[gpuID], CL_DEVICE_NAME, 128, (void *)&buff[0], NULL);
	fprintf(stderr, "Selected Device %d: %s\n", gpuID, buff);

	//Now that we've picked a reasonable default, fill in the details for the frame object
	accelOpenCLInitStackFrame(frame, gpuID); 

	return(ret);
}


//  !this kernel works for 3D data only.
//  ! PHI1 and PHI2 are input arrays.
//  ! s* parameters are start values in each dimension.
//  ! e* parameters are end values in each dimension.
//  ! s* and e* are only necessary when the halo layers 
//  !   has different thickness along various directions.
//  ! i,j,k are the array dimensions
//  ! len_ is number of threads in a threadblock.
//  !      This can be computed in the kernel itself.
cl_int opencl_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], void * data1, void * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, accel_type_id type, int async, cl_event * event) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_len =  (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	//before enqueuing, get a copy of the top stack frame
	accelOpenCLStackFrame * frame = accelOpenCLTopStackFrame();

	switch (type) {
		case a_db:
			kern = frame->kernel_dotProd_db;
			break;

		case a_fl:
			kern = frame->kernel_dotProd_fl;
			break;

		case a_ul:
			kern = frame->kernel_dotProd_ul;
			break;

		case a_in:
			kern = frame->kernel_dotProd_in;
			break;

		case a_ui:
			kern = frame->kernel_dotProd_ui;
			break;

		default:
			fprintf(stderr, "Error: Function 'opencl_dotProd' not implemented for selected type!\n");
			return -1;
			break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret =  clSetKernelArg(kern, 0, sizeof(cl_mem *), &data1);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &data2);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*array_size)[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*array_size)[1]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*array_size)[2]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*arr_start)[0]);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_int), &(*arr_start)[1]);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_int), &(*arr_start)[2]);
	ret |= clSetKernelArg(kern, 8, sizeof(cl_int), &(*arr_end)[0]);
	ret |= clSetKernelArg(kern, 9, sizeof(cl_int), &(*arr_end)[1]);
	ret |= clSetKernelArg(kern, 10, sizeof(cl_int), &(*arr_end)[2]);
	ret |= clSetKernelArg(kern, 11, sizeof(cl_int), &(*grid_size)[2]);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_mem *), &reduced_val);
	ret |= clSetKernelArg(kern, 13, sizeof(cl_int), &smem_len);
	switch (type) {
		case a_db:
			ret |= clSetKernelArg(kern, 14, smem_len*sizeof(cl_double), NULL);
			break;

		case a_fl:
			ret |= clSetKernelArg(kern, 14, smem_len*sizeof(cl_float), NULL);
			break;

		case a_ul:
			ret |= clSetKernelArg(kern, 14, smem_len*sizeof(cl_ulong), NULL);
			break;

		case a_in:
			ret |= clSetKernelArg(kern, 14, smem_len*sizeof(cl_int), NULL);
			break;

		case a_ui:
			ret |= clSetKernelArg(kern, 14, smem_len*sizeof(cl_uint), NULL);
			break;

		//Shouldn't be reachable, but cover our bases
		default:
			fprintf(stderr, "Error: unexpected type, cannot set shared memory size in 'opencl_dotProd'!\n");
	}
		
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0, NULL, event);
	
	//TODO find a way to make explicit sync optional
	if (!async) ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return(ret);
}


cl_int opencl_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], void * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, accel_type_id type, int async, cl_event * event) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_len =  (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};

	//before enqueuing, get a copy of the top stack frame
	accelOpenCLStackFrame * frame = accelOpenCLTopStackFrame();
	
	switch (type) {
		case a_db:
			kern = frame->kernel_reduce_db;
			break;

		case a_fl:
			kern = frame->kernel_reduce_fl;
			break;

		case a_ul:
			kern = frame->kernel_reduce_ul;
			break;

		case a_in:
			kern = frame->kernel_reduce_in;
			break;

		case a_ui:
			kern = frame->kernel_reduce_ui;
			break;

		default:
			fprintf(stderr, "Error: Function 'opencl_reduce' not implemented for selected type!\n");
			return -1;
			break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret =  clSetKernelArg(kern, 0, sizeof(cl_mem *), &data);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_int), &(*array_size)[0]);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*array_size)[1]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*array_size)[2]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*arr_start)[0]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*arr_start)[1]);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_int), &(*arr_start)[2]);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_int), &(*arr_end)[0]);
	ret |= clSetKernelArg(kern, 8, sizeof(cl_int), &(*arr_end)[1]);
	ret |= clSetKernelArg(kern, 9, sizeof(cl_int), &(*arr_end)[2]);
	ret |= clSetKernelArg(kern, 10, sizeof(cl_int), &(*grid_size)[2]);
	ret |= clSetKernelArg(kern, 11, sizeof(cl_mem *), &reduced_val);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_int), &smem_len);
	switch (type) {
		case a_db:
			ret |= clSetKernelArg(kern, 13, smem_len*sizeof(cl_double), NULL);
			break;

		case a_fl:
			ret |= clSetKernelArg(kern, 13, smem_len*sizeof(cl_float), NULL);
			break;

		case a_ul:
			ret |= clSetKernelArg(kern, 13, smem_len*sizeof(cl_ulong), NULL);
			break;

		case a_in:
			ret |= clSetKernelArg(kern, 13, smem_len*sizeof(cl_int), NULL);
			break;

		case a_ui:
			ret |= clSetKernelArg(kern, 13, smem_len*sizeof(cl_uint), NULL);
			break;

		//Shouldn't be reachable, but cover our bases
		default:
			fprintf(stderr, "Error: unexpected type, cannot set shared memory size in 'opencl_reduce'!\n");
	}
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0, NULL, event);
	
	//TODO find a way to make explicit sync optional
	if (!async) ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return(ret);
}

cl_int opencl_transpose_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void *outdata, size_t (* dim_xy)[3], accel_type_id type, int async, cl_event * event) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_len =  (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
// TODO update to use user provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	size_t grid[3] = {((*dim_xy)[0]+TRANSPOSE_TILE_DIM-1)/TRANSPOSE_TILE_DIM, ((*dim_xy)[1]+TRANSPOSE_TILE_DIM-1)/TRANSPOSE_TILE_DIM, 1};
	size_t block[3] = {TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, 1};
	//TODO as the frame grows larger with more kernels, this overhead will start to add up
	// Need a better (safe) way of accessing the stack for kernel launches
	//before enqueuing, get a copy of the top stack frame
	accelOpenCLStackFrame * frame = accelOpenCLTopStackFrame();
	
	switch (type) {
		case a_db:
			kern = frame->kernel_transpose_2d_face_db;
			break;

		case a_fl:
			kern = frame->kernel_transpose_2d_face_fl;
			break;

		case a_ul:
			kern = frame->kernel_transpose_2d_face_ul;
			break;

		case a_in:
			kern = frame->kernel_transpose_2d_face_in;
			break;

		case a_ui:
			kern = frame->kernel_transpose_2d_face_ui;
			break;

		default:
			fprintf(stderr, "Error: Function 'opencl_transpose_2d_face' not implemented for selected type!\n");
			return -1;
			break;

	}
	ret =  clSetKernelArg(kern, 0, sizeof(cl_mem *), &outdata);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &indata);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*dim_xy)[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*dim_xy)[1]);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 2, NULL, grid, block, 0, NULL, event);
	
	//TODO find a way to make explicit sync optional
	if (!async) ret |= clFinish(frame->queue);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return(ret);
}

cl_int opencl_pack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, accel_2d_face_indexed *face, int *remain_dim, accel_type_id type, int async, cl_event * event_k1, cl_event * event_c1, cl_event *event_c2, cl_event *event_c3) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_size =  face->count*256*sizeof(int);
	//before enqueuing, get a copy of the top stack frame
	accelOpenCLStackFrame * frame = accelOpenCLTopStackFrame();

	//copy required pieces of the face struct into constant memory
	ret = clEnqueueWriteBuffer(frame->queue, frame->constant_face_size, ((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int)*face->count, face->size, 0, NULL, event_c1);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_stride, ((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int)*face->count, face->stride, 0, NULL, event_c2);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_child_size, ((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int)*face->count, remain_dim, 0, NULL, event_c2);
//TODO update to use user-provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	size_t grid[3] = {(remain_dim[0]+256-1)/256, 1, 1};
	size_t block[3] = {256, 1, 1};
//TODO Timing needs to be made consistent with CUDA (ie the event should return time for copying to constant memory and the kernel
	
	switch (type) {
		case a_db:
			kern = frame->kernel_pack_2d_face_db;
			break;

		case a_fl:
			kern = frame->kernel_pack_2d_face_fl;
			break;

		case a_ul:
			kern = frame->kernel_pack_2d_face_ul;
			break;

		case a_in:
			kern = frame->kernel_pack_2d_face_in;
			break;

		case a_ui:
			kern = frame->kernel_pack_2d_face_ui;
			break;

		default:
			fprintf(stderr, "Error: Function 'opencl_pack_2d_face' not implemented for selected type!\n");
			return -1;
			break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret =  clSetKernelArg(kern, 0, sizeof(cl_mem *), &packed_buf);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &buf);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &remain_dim[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(face->start));
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(face->count));
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, grid, block, 0, NULL, event_k1);
	
	//TODO find a way to make explicit sync optional
	if (!async) ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return(ret);
}

cl_int opencl_unpack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, accel_2d_face_indexed *face, int *remain_dim, accel_type_id type, int async, cl_event * event_k1, cl_event * event_c1, cl_event *event_c2, cl_event *event_c3) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_size =  face->count*256*sizeof(int);
	//before enqueuing, get a copy of the top stack frame
	accelOpenCLStackFrame * frame = accelOpenCLTopStackFrame();

	//copy required pieces of the face struct into constant memory
	ret = clEnqueueWriteBuffer(frame->queue, frame->constant_face_size, ((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int)*face->count, face->size, 0, NULL, event_c1);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_stride, ((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int)*face->count, face->stride, 0, NULL, event_c2);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_child_size, ((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int)*face->count, remain_dim, 0, NULL, event_c2);
//TODO update to use user-provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	size_t grid[3] = {(remain_dim[0]+256-1)/256, 1, 1};
	size_t block[3] = {256, 1, 1};
//TODO Timing needs to be made consistent with CUDA (ie the event should return time for copying to constant memory and the kernel
	
	switch (type) {
		case a_db:
			kern = frame->kernel_unpack_2d_face_db;
			break;

		case a_fl:
			kern = frame->kernel_unpack_2d_face_fl;
			break;

		case a_ul:
			kern = frame->kernel_unpack_2d_face_ul;
			break;

		case a_in:
			kern = frame->kernel_unpack_2d_face_in;
			break;

		case a_ui:
			kern = frame->kernel_unpack_2d_face_ui;
			break;

		default:
			fprintf(stderr, "Error: Function 'opencl_unpack_2d_face' not implemented for selected type!\n");
			return -1;
			break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret =  clSetKernelArg(kern, 0, sizeof(cl_mem *), &packed_buf);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &buf);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &remain_dim[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(face->start));
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(face->count));
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, grid, block, 0, NULL, event_k1);
	
	//TODO find a way to make explicit sync optional
	if (!async) ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return(ret);
}
