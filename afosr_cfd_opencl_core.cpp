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
(*frame)->platform = t->frame.platform;
(*frame)->device = t->frame.device;
(*frame)->context = t->frame.context;
(*frame)->queue = t->frame.queue;
(*frame)->program_opencl_core = t->frame.program_opencl_core;
(*frame)->kernel_reduction3 = t->frame.kernel_reduction3;

//This should be the end of the hazards;
}

//ASSUME HAZARD POINTERS ARE ALREADY SET FOR node BY THE CALLING METHOD
void copyStackFrameToNode(accelOpenCLStackFrame * f, accelOpenCLStackNode ** node) {

(*node)->frame.platform = f->platform;
(*node)->frame.device = f->device;
(*node)->frame.context = f->context;
(*node)->frame.queue = f->queue;
(*node)->frame.program_opencl_core = f->program_opencl_core;
(*node)->frame.kernel_reduction3 = f->kernel_reduction3;

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

//TODO add intrinsics for non-GCC compilers
//while (old != __sync_val_compare_and_swap(&CLStack, old, newNode)) {
	//must have been changed between reading and swapping, get the new top
//	old = CLStack;
	//update the new node's next pointer to whatever the new top is
//	newNode->next = old;
//}
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
	clBuildProgram((*frame)->program_opencl_core, 1, &((*frame)->device), "-I . -g -s\"/home/psath/private_repos/afosr_cfd_lib/afosr_cfd_opencl_core.cl\"", NULL, NULL);

//Stub to get build log
size_t logsize = 0;
clGetProgramBuildInfo((*frame)->program_opencl_core, (*frame)->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
char * log = (char *) malloc (sizeof(char) *(logsize+1));
clGetProgramBuildInfo((*frame)->program_opencl_core, (*frame)->device, CL_PROGRAM_BUILD_LOG, logsize, log, NULL);
fprintf(stderr, "CL_PROGRAM_BUILD_LOG:\n%s", log);
free(log);
	(*frame)->kernel_reduction3 = clCreateKernel((*frame)->program_opencl_core, "kernel_reduction3", NULL);

}

//calls all the necessary CLRelease* calls for frame members
//DOES NOT:
//	pop any stack nodes
//	free any stack nodes
//	free program source
//	implement any thread safety, frames should always be thread private, only the stack should be shared, and all franes must be copied to or from stack nodes using the hazard-aware copy methods.
//	 (more specifically, copying a frame to a node doesn't need to be hazard-aware, as the node cannot be shared unless copied inside the hazard-aware accelOpenCLPushStackFrame. Pop, Top, and copyStackNodeToFrame are all hazard aware and provide a thread-private copy back to the caller.)
cl_int accelOpenCLDestroyStackFrame(accelOpenCLStackFrame * frame) {

clReleaseKernel(frame->kernel_reduction3);
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


   //end subroutine block_reduction

//Paul - Implementation of double atomicAdd from CUDA Programming Guide: Appendix B.12


//  !this kernel works for 3D data only.
//  ! PHI1 and PHI2 are input arrays.
//  ! s* parameters are start values in each dimension.
//  ! e* parameters are end values in each dimension.
//  ! s* and e* are only necessary when the halo layers 
//  !   has different thickness along various directions.
//  ! i,j,k are the array dimensions
//  ! len_ is number of threads in a threadblock.
//  !      This can be computed in the kernel itself.
//  ATTRIBUTES(GLOBAL) &
//     & SUBROUTINE KERNEL_REDUCTION3(PHI1,PHI2,j,i,k,sx,sy,sz,ex,ey,ez,gz,&
//     & REDUCTION,len_)
	


cl_int opencl_dotProd_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], double * data1, double * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val) {
	cl_int ret;
	cl_int smem_len =  (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	printf("SMEM: %d\n", smem_len);

	//before enqueuing, get a copy of the top stack frame
	accelOpenCLStackFrame * frame = accelOpenCLTopStackFrame();

/*CU2CL Note -- Inserted temporary variable for kernel literal argument 13!*/
ret =	clSetKernelArg(frame->kernel_reduction3, 0, sizeof(cl_mem *), &data1);
ret |= clSetKernelArg(frame->kernel_reduction3, 1, sizeof(cl_mem *), &data2);
ret |= clSetKernelArg(frame->kernel_reduction3, 2, sizeof(cl_int), &(*array_size)[0]);
ret |= clSetKernelArg(frame->kernel_reduction3, 3, sizeof(cl_int), &(*array_size)[1]);
ret |= clSetKernelArg(frame->kernel_reduction3, 4, sizeof(cl_int), &(*array_size)[2]);
ret |= clSetKernelArg(frame->kernel_reduction3, 5, sizeof(cl_int), &(*arr_start)[0]);
ret |= clSetKernelArg(frame->kernel_reduction3, 6, sizeof(cl_int), &(*arr_start)[1]);
ret |= clSetKernelArg(frame->kernel_reduction3, 7, sizeof(cl_int), &(*arr_start)[2]);
ret |= clSetKernelArg(frame->kernel_reduction3, 8, sizeof(cl_int), &(*arr_end)[1]);
ret |= clSetKernelArg(frame->kernel_reduction3, 9, sizeof(cl_int), &(*arr_end)[0]);
ret |= clSetKernelArg(frame->kernel_reduction3, 10, sizeof(cl_int), &(*arr_end)[2]);
ret |= clSetKernelArg(frame->kernel_reduction3, 11, sizeof(cl_int), &(*grid_size)[2]);
ret |= clSetKernelArg(frame->kernel_reduction3, 12, sizeof(cl_mem *), &reduced_val);
ret |= clSetKernelArg(frame->kernel_reduction3, 13, sizeof(cl_int), &smem_len);
ret |= clSetKernelArg(frame->kernel_reduction3, 14, smem_len*sizeof(cl_double), NULL);
ret |= clEnqueueNDRangeKernel(frame->queue, frame->kernel_reduction3, 3, NULL, grid, block, 0, NULL, NULL);
	//kernel_reduction3<<<grid,block,smem_size>>>(data1, data2, (*array_size)[0], (*array_size)[1], (*array_size)[2], (*arr_start)[0], (*arr_start)[1], (*arr_start)[2], (*arr_end)[1], (*arr_end)[0], (*arr_end)[2], 8, reduced_val, 8*8*8);
	ret |= clFinish(frame->queue);
	printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members

	return(ret);
}
