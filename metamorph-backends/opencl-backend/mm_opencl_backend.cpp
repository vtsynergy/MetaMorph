/** \file
 * OpenCL backend function implementations
 * \bug OpenCL kernel wrappers need to be modified to copy primitive type variables into their cl_\<type\> equivalents before clSetKernelArg to ensure width between device and host
 */


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../../metamorph-backends/opencl-backend/mm_opencl_backend.h"

#if defined(__OPENCLCC__) || defined(__cplusplus)
extern "C" {
#endif

/** The globally-exposed cl_context for the most recently initialized OpenCL frame */
cl_context meta_context;
/** The globally-exposed cl_command_queue for the most recently initialized OpenCL frame */
cl_command_queue meta_queue;
/** The globally-exposed cl_device_id for the most recently initialized OpenCL frame */
cl_device_id meta_device;

//Warning, none of these variables are threadsafe, so only one thread should ever
// perform the one-time scan!!
cl_uint num_platforms, num_devices;
cl_platform_id * __meta_platforms_array = NULL;
cl_device_id * __meta_devices_array = NULL;

/**
 * Simple data structure to separate the user-exposed concept of OpenCL frames from their storage implementation
 */
typedef struct metaOpenCLStackNode {
	metaOpenCLStackFrame frame; /** The frame to store */
	struct metaOpenCLStackNode * next; /** pointer to the previously-initialized frame */
} metaOpenCLStackNode;

metaOpenCLStackNode * CLStack = NULL;

/**
 * Function to lookup a specific device across all platforms by its name string
 * \param desired a NULL-terminated string corresponsind to the exact name of the desired device as returned by OpenCL APIs
 * \param devices a pointer to an array of devices of length numDevices to search for the named device
 * \param numDevices the length of the devices array to search
 * \return The index in devices of the first device found with a name matching desired, or -1 if not found
 */
int metaOpenCLGetDeviceID(char * desired, cl_device_id * devices,
		int numDevices) {
	char buff[128];
	int i;
	for (i = 0; i < numDevices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, (void *) buff, NULL);
		//printf("%s\n", buff);
		if (strcmp(desired, buff) == 0)
			return i;
	}
	return -1;
}

/**
 * Basic one-time scan of all platforms to assemble the internal devices array.
 * \warning This is not threadsafe, and I'm not sure we'll ever make it safe.
 */
void metaOpenCLQueryDevices() {
	int i;
	num_platforms = 0, num_devices = 0;
	cl_uint temp_uint, temp_uint2;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS)
		printf("Failed to query platform count!\n");
	printf("Number of OpenCL Platforms: %d\n", num_platforms);

	__meta_platforms_array = (cl_platform_id *) malloc(
			sizeof(cl_platform_id) * (num_platforms + 1));

	if (clGetPlatformIDs(num_platforms, &__meta_platforms_array[0], NULL) != CL_SUCCESS)
		printf("Failed to get platform IDs\n");

	for (i = 0; i < num_platforms; i++) {
		temp_uint = 0;
		fprintf(stderr,
				"OCL DEBUG: clGetDeviceIDs Count query on platform[%d] has address[%x]!\n",
				i, &temp_uint);
		if (clGetDeviceIDs(__meta_platforms_array[i], CL_DEVICE_TYPE_ALL, 0, NULL,
				&temp_uint) != CL_SUCCESS)
			printf("Failed to query device count on platform %d!\n", i);
		num_devices += temp_uint;
	}
	printf("Number of Devices: %d\n", num_devices);

	__meta_devices_array = (cl_device_id *) malloc(sizeof(cl_device_id) * (num_devices + 1));
	temp_uint = 0;
	for (i = 0; i < num_platforms; i++) {
		fprintf(stderr,
				"OCL DEBUG: clGetDeviceIDs IDs query on platform[%d] has addresses[%x][%x]!\n",
				i, &__meta_devices_array[temp_uint], &temp_uint2);
		if (clGetDeviceIDs(__meta_platforms_array[i], CL_DEVICE_TYPE_ALL, num_devices,
				&__meta_devices_array[temp_uint], &temp_uint2) != CL_SUCCESS)
			printf("Failed to query device IDs on platform %d!\n", i);
		temp_uint += temp_uint2;
		temp_uint2 = 0;
	}

	//TODO figure out somewhere else to put this, like a terminating callback or something...
	//free(__meta_devices_array);
	//free(__meta_platforms_array);
}

//Returns the size of the first program with corresponding name found in METAMORPH_OCL_KERNEL_PATH
//If none is found, returns -1. Client responsible for handling non-existent kernels gracefully.
size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc) {
	//Construct the path string
	//environment variable + METAMORPH_OCL_KERNEL_PATH
	char * path = NULL;
	if (getenv("METAMORPH_OCL_KERNEL_PATH") != NULL) {
		size_t needed = snprintf(NULL, 0, "%s:" METAMORPH_OCL_KERNEL_PATH, getenv("METAMORPH_OCL_KERNEL_PATH"));
		path = (char*)calloc(needed+1, sizeof(char));
		snprintf(path, needed+1, "%s:" METAMORPH_OCL_KERNEL_PATH, getenv("METAMORPH_OCL_KERNEL_PATH"));
	} else {
		size_t needed = snprintf(NULL, 0, METAMORPH_OCL_KERNEL_PATH);
		path = (char*)calloc(needed+1, sizeof(char));
		snprintf(path, needed+1, METAMORPH_OCL_KERNEL_PATH);
	}
	char const * token = strtok(path, ":");
	FILE *f = NULL;
	if (token == NULL) token = "."; //handle completely empty path
	//loop over all paths until a copy is found, notify if none found
	while (token != NULL && f == NULL) {
		if ((strcmp(token, "") == 0)) token = "."; //handle empty token
		size_t abs_path_sz = snprintf(NULL, 0, "%s/%s", token, filename);
		char * abs_path = (char*)calloc(abs_path_sz+1, sizeof(char));
		snprintf(abs_path, abs_path_sz+1, "%s/%s", token, filename);
		f = fopen(abs_path, "r");
		token = strtok(NULL, ":");
	}
	//TODO if none is found, how to handle? Don't need to crash the program, we can just not allow the kernel(s) in this file to run
	if (f == NULL) {
		fprintf(stderr, "MetaMorph could not find kernel file \"%s\",subsequent kernel launches will return CL_INVALID_PROGRAM (%d)\n", filename, CL_INVALID_PROGRAM);
		return -1;
	}

	fseek(f, 0, SEEK_END);
	size_t len = (size_t) ftell(f);
	*progSrc = (const char *) malloc(sizeof(char) * len);
	rewind(f);
	fread((void *) *progSrc, len, 1, f);
	fclose(f);
	return len;
}


/**
 * Build a single cl_program for the provided frame with the provided arguments
 * \param frame the frame to build the program for
 * \param prog A pointer to the cl_program to build (should be an offset into frame)
 * \param prog_args The build arguments to use to build the program
 * \warning kernel-, device-, or configuration-specific build args may be prepended to the provided prog_args
 * \return an OpenCL error code if anything went wrong, or a CL_SUCCESS otherwise
 */
cl_int metaOpenCLBuildSingleKernelProgram(metaOpenCLStackFrame * frame, cl_program * prog, char const * prog_args) {
	if (prog_args == NULL) prog_args = "";
	cl_int ret = CL_SUCCESS;
	char * args = NULL;
	//FIXME Do these OpenCL build strings actually need "-D OPENCL" when building for WITH_INTELFPGA?
	if (getenv("METAMORPH_MODE") != NULL) {
		if (strcmp(getenv("METAMORPH_MODE"), "OpenCL") == 0) {
			size_t needed =
					snprintf(NULL, 0,
							"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
							TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
			args = (char *) malloc(needed);
			snprintf(args, needed,
					"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
					TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
		} else if (strcmp(getenv("METAMORPH_MODE"), "OpenCL_DEBUG") == 0) {
			size_t needed =
					snprintf(NULL, 0,
							"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) -g -cl-opt-disable %s ",
							TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
			args = (char *) malloc(needed);
			snprintf(args, needed,
					"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) -g -cl-opt-disable %s ",
					TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
		}
	} else {
		//Do the same as if METAMORPH_MODE was set as OpenCL
		size_t needed =
				snprintf(NULL, 0,
						"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
						TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
		args = (char *) malloc(needed);
		snprintf(args, needed,
				"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
				TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);

	}
	ret |= clBuildProgram(*prog, 1, &(frame->device), args,
			NULL, NULL);
	//Let us know if there's any errors in the build process
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "Error in clBuildProgram: %d!\n", ret);
		ret = CL_SUCCESS;
	}
	//Stub to get build log
	size_t logsize = 0;
	clGetProgramBuildInfo(*prog, frame->device,
			CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
	char * log = (char *) malloc(sizeof(char) * (logsize + 1));
	clGetProgramBuildInfo(*prog, frame->device,
			CL_PROGRAM_BUILD_LOG, logsize, log, NULL);
	if (logsize > 2) fprintf(stderr, "CL_PROGRAM_BUILD_LOG:\n%s", log);
	free(log);
	return ret;
}

/**
 * For the provided frame, read, create, and build all necessary programs and kernels
 * \param frame The frame to initialize program(s) and kernels for
 * \return an OpenCL error code if anything went wrong, otherwise CL_SUCCESS
 */
cl_int metaOpenCLBuildProgram(metaOpenCLStackFrame * frame) {
	cl_int ret = CL_SUCCESS;
#if defined(WITH_INTELFPGA) && defined(OPENCL_SINGLE_KERNEL_PROGS)
	//will need a separate buffer for each kernel

#define ENSURE_SRC(name) if (frame->metaCLbinLen_##name == 0) \
	frame->metaCLbinLen_##name = metaOpenCLLoadProgramSource("mm_opencl_intelfpga_backend_"##name".aocx", &(frame->metaCLbin_##name)); \
	} \
	if (frame->metaCLbinLen_##name != -1) frame->program_##name = clCreateProgramWithBinary(frame->context, 1, &(frame->device), &(frame->metaCLbinLen_##name), (const unsigned char **) &(frame->metaCLbin_##name), NULL, NULL);

	ENSURE_SRC(reduce_db);
	ENSURE_SRC(reduce_fl);
	ENSURE_SRC(reduce_ul);
	ENSURE_SRC(reduce_in);
	ENSURE_SRC(reduce_ui);
	ENSURE_SRC(dotProd_db);
	ENSURE_SRC(dotProd_fl);
	ENSURE_SRC(dotProd_ul);
	ENSURE_SRC(dotProd_in);
	ENSURE_SRC(dotProd_ui);
	ENSURE_SRC(transpose_2d_face_db);
	ENSURE_SRC(transpose_2d_face_fl);
	ENSURE_SRC(transpose_2d_face_ul);
	ENSURE_SRC(transpose_2d_face_in);
	ENSURE_SRC(transpose_2d_face_ui);
	ENSURE_SRC(pack_2d_face_db);
	ENSURE_SRC(pack_2d_face_fl);
	ENSURE_SRC(pack_2d_face_ul);
	ENSURE_SRC(pack_2d_face_in);
	ENSURE_SRC(pack_2d_face_ui);
	ENSURE_SRC(unpack_2d_face_db);
	ENSURE_SRC(unpack_2d_face_fl);
	ENSURE_SRC(unpack_2d_face_ul);
	ENSURE_SRC(unpack_2d_face_in);
	ENSURE_SRC(unpack_2d_face_ui);
	ENSURE_SRC(stencil_3d7p_db);
	ENSURE_SRC(stencil_3d7p_fl);
	ENSURE_SRC(stencil_3d7p_ul);
	ENSURE_SRC(stencil_3d7p_in);
	ENSURE_SRC(stencil_3d7p_ui);
	ENSURE_SRC(csr_db);
	ENSURE_SRC(csr_fl);
	ENSURE_SRC(csr_ul);
	ENSURE_SRC(csr_in);
	ENSURE_SRC(csr_ui);
//	ENSURE_SRC(crc_db);
//	ENSURE_SRC(crc_fl);
//	ENSURE_SRC(crc_ul);
//	ENSURE_SRC(crc_in);
	ENSURE_SRC(crc_ui);
#undef ENSURE_SRC
#elif defined(WITH_INTELFPGA) && !defined(OPENCL_SINGLE_KERNEL_PROGS)
	if (frame->metaCLProgLen == 0) {
		frame->metaCLProgLen = metaOpenCLLoadProgramSource("mm_opencl_intelfpga_backend.aocx", &(frame->metaCLProgSrc));
	}
	if (frame->metaCLProgLen != -1) frame->program_opencl_core=clCreateProgramWithBinary(frame->context,1,&(frame->device), &(frame->metaCLProgLen),(const unsigned char**) &(frame->metaCLProgSrc),NULL,NULL);
#elif !defined(WITH_INTELFPGA) && defined(OPENCL_SINGLE_KERNEL_PROGS)
	if (frame->metaCLProgLen == 0) {
		frame->metaCLProgLen = metaOpenCLLoadProgramSource("mm_opencl_backend.cl",&(frame->metaCLProgSrc));
	}
	//They use the same source with different defines
	if (frame->metaCLProgLen != -1) {
		frame->program_reduce_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_reduce_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_reduce_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_reduce_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_reduce_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_dotProd_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_dotProd_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_dotProd_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_dotProd_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_dotProd_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_transpose_2d_face_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_transpose_2d_face_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_transpose_2d_face_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_transpose_2d_face_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_transpose_2d_face_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_pack_2d_face_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_pack_2d_face_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_pack_2d_face_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_pack_2d_face_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_pack_2d_face_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_unpack_2d_face_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_unpack_2d_face_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_unpack_2d_face_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_unpack_2d_face_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_unpack_2d_face_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_stencil_3d7p_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_stencil_3d7p_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_stencil_3d7p_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_stencil_3d7p_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_stencil_3d7p_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_csr_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_csr_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_csr_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_csr_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_csr_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//		frame->program_crc_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//		frame->program_crc_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//		frame->program_crc_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//		frame->program_crc_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
		frame->program_crc_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	}
//TODO Implement (or can this be the same as OpenCL without IntelFPGA, since it's just reading the source and the defines come in during build
#else
	if (frame->metaCLProgLen == 0) {
		frame->metaCLProgLen = metaOpenCLLoadProgramSource("mm_opencl_backend.cl",
				&(frame->metaCLProgSrc));
	}
	if (frame->metaCLProgLen != -1) frame->program_opencl_core = clCreateProgramWithSource(frame->context, 1,
			&(frame->metaCLProgSrc), &(frame->metaCLProgLen), &ret);
#endif

//TODO Support OPENCL_SINGLE_KERNEL_PROGS
#ifndef OPENCL_SINGLE_KERNEL_PROGS
	ret |= metaOpenCLBuildSingleKernelProgram(frame, &frame->program_opencl_core, NULL);
	if (frame->metaCLProgLen != -1) {
		frame->kernel_reduce_db = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_db", &ret);
		frame->kernel_reduce_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_fl", &ret);
		frame->kernel_reduce_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_ul", &ret);
		frame->kernel_reduce_in = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_in", &ret);
		frame->kernel_reduce_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_ui", &ret);
		frame->kernel_dotProd_db = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_db", &ret);
		frame->kernel_dotProd_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_fl", &ret);
		frame->kernel_dotProd_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_ul", &ret);
		frame->kernel_dotProd_in = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_in", &ret);
		frame->kernel_dotProd_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_ui", &ret);
		frame->kernel_transpose_2d_face_db = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_db", &ret);
		frame->kernel_transpose_2d_face_fl = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_fl", &ret);
		frame->kernel_transpose_2d_face_ul = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_ul", &ret);
		frame->kernel_transpose_2d_face_in = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_in", &ret);
		frame->kernel_transpose_2d_face_ui = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_ui", &ret);
		frame->kernel_pack_2d_face_db = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_db", &ret);
		frame->kernel_pack_2d_face_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_fl", &ret);
		frame->kernel_pack_2d_face_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_ul", &ret);
		frame->kernel_pack_2d_face_in = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_in", &ret);
		frame->kernel_pack_2d_face_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_ui", &ret);
		frame->kernel_unpack_2d_face_db = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_db", &ret);
		frame->kernel_unpack_2d_face_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_fl", &ret);
		frame->kernel_unpack_2d_face_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_ul", &ret);
		frame->kernel_unpack_2d_face_in = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_in", &ret);
		frame->kernel_unpack_2d_face_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_ui", &ret);
		frame->kernel_stencil_3d7p_db = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_db", &ret);
		frame->kernel_stencil_3d7p_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_fl", &ret);
		frame->kernel_stencil_3d7p_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_ul", &ret);
		frame->kernel_stencil_3d7p_in = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_in", &ret);
		frame->kernel_stencil_3d7p_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_ui", &ret);
		frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_db", &ret);
		frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_fl", &ret);
		frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_ul", &ret);
		frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_in", &ret);
		frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_ui", &ret);
//	frame->kernel_crc_db = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_db", &ret);
//	frame->kernel_crc_fl = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_fl", &ret);
//	frame->kernel_crc_ul = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_ul", &ret);
//	frame->kernel_crc_in = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_in", &ret);
		frame->kernel_crc_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_crc_ui", &ret);
	}
#else
#ifdef WITH_INTELFPGA
#define BUILD_PROG_AND_KERNEL(name, opts) ret |= metaOpenCLBuildSingleKernelProgram(frame, &frame->program_##name, opts); \
	if (frame->metaCLbinLen_##name != -1) frame->kernel_##name = clCreateKernel(frame->program_##name, "kernel_"#name, &ret); \
        if (ret != CL_SUCCESS) fprintf(stderr, "Error in clCreateKernel for kernel_"#name" %d\n", ret);
#else
#define BUILD_PROG_AND_KERNEL(name, opts) ret |= metaOpenCLBuildSingleKernelProgram(frame, &frame->program_##name, opts); \
	if (frame->metaCLProgLen != -1) frame->kernel_##name = clCreateKernel(frame->program_##name, "kernel_"#name, &ret); \
        if (ret != CL_SUCCESS) fprintf(stderr, "Error in clCreateKernel for kernel_"#name" %d\n", ret);
#endif
//	ret |= metaOpenCLBuildSingleKernelProgram(frame, frame->program_reduce_db, "-D DOUBLE -D KERNEL_REDUCE");
//	frame->kernel_reduce_db = clCreateKernel(frame->program_opencl_reduce_db, "kernel_reduce_db", &ret);
	BUILD_PROG_AND_KERNEL(reduce_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(dotProd_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(pack_2d_face_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(csr_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CSR")
//	BUILD_PROG_AND_KERNEL(crc_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_CRC")
//	BUILD_PROG_AND_KERNEL(crc_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_CRC")
//	BUILD_PROG_AND_KERNEL(crc_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_CRC")
//	BUILD_PROG_AND_KERNEL(crc_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_CRC")
	BUILD_PROG_AND_KERNEL(crc_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CRC")
#endif
	frame->kernels_init=1;
        //Reinitialize OpenCL add-on modules
        meta_reinitialize_modules(module_implements_opencl);
	return ret;
}

//This should not be exposed to the user, just the top and pop functions built on top of it
//for thread-safety, returns a new metaOpenCLStackNode, which is a direct copy
// of the top at some point in time.
// this way, the user can still use top, without having to manage hazard pointers themselves
/**
 * Internally copy whatever's on top of the stack to a frame that is safe to hand to users.
 * Never give direct access to the stack so that it can one day be made threadsafe via hazard pointers
 * \param t The Internal stack node to copy
 * \param frame The address of a frame * variable to return the newly allocated and copied frame in
 * \warning ASSUME HAZARD POINTERS ARE ALREADY SET FOR t BY THE CALLING METHOD
 */
void copyStackNodeToFrame(metaOpenCLStackNode * t,
		metaOpenCLStackFrame ** frame) {

	//From here out, we have hazards
	//Copy all the parameters - REALLY HAZARDOUS

	*(*frame) = t->frame;

	//Top-level context info
/*
	(*frame)->platform = t->frame.platform;
	(*frame)->device = t->frame.device;
	(*frame)->context = t->frame.context;
	(*frame)->queue = t->frame.queue;
#if !defined(OPENCL_SINGLE_KERNEL_PROGS) || !defined(WITH_INTELFPGA)
	//copy just the regular buffer and length
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
#else
	//TODO copy all the buffers and lengths
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
#endif
#ifdef (OPENCL_SINGLE_KERNEL_PROGS)
	(*frame)->program_opencl_core = t->frame.program_opencl_core;
	//TODO copy just the regular buffer and length
#else
//TODO copy extra cl_programs
#endif

//TODO this should be acheivable with a memcpy
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
	(*frame)->kernel_transpose_2d_face_db =
			t->frame.kernel_transpose_2d_face_db;
	(*frame)->kernel_transpose_2d_face_fl =
			t->frame.kernel_transpose_2d_face_fl;
	(*frame)->kernel_transpose_2d_face_ul =
			t->frame.kernel_transpose_2d_face_ul;
	(*frame)->kernel_transpose_2d_face_in =
			t->frame.kernel_transpose_2d_face_in;
	(*frame)->kernel_transpose_2d_face_ui =
			t->frame.kernel_transpose_2d_face_ui;
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
	(*frame)->kernel_stencil_3d7p_db = t->frame.kernel_stencil_3d7p_db;
	(*frame)->kernel_stencil_3d7p_fl = t->frame.kernel_stencil_3d7p_fl;
	(*frame)->kernel_stencil_3d7p_ul = t->frame.kernel_stencil_3d7p_ul;
	(*frame)->kernel_stencil_3d7p_in = t->frame.kernel_stencil_3d7p_in;
	(*frame)->kernel_stencil_3d7p_ui = t->frame.kernel_stencil_3d7p_ui;
	(*frame)->kernel_csr_db = t->frame.kernel_csr_db;
	(*frame)->kernel_csr_fl = t->frame.kernel_csr_fl;
	(*frame)->kernel_csr_ul = t->frame.kernel_csr_ul;
	(*frame)->kernel_csr_in = t->frame.kernel_csr_in;
	(*frame)->kernel_csr_ui = t->frame.kernel_csr_ui;
	(*frame)->kernel_crc_db = t->frame.kernel_crc_db;
	(*frame)->kernel_crc_fl = t->frame.kernel_crc_fl;
	(*frame)->kernel_crc_ul = t->frame.kernel_crc_ul;
	(*frame)->kernel_crc_in = t->frame.kernel_crc_in;
	(*frame)->kernel_crc_ui = t->frame.kernel_crc_ui;

	//Internal buffers
	(*frame)->constant_face_size = t->frame.constant_face_size;
	(*frame)->constant_face_stride = t->frame.constant_face_stride;
	(*frame)->constant_face_child_size = t->frame.constant_face_child_size;
	(*frame)->red_loc = t->frame.red_loc;
	//This should be the end of the hazards;
*/
}

//ASSUME HAZARD POINTERS ARE ALREADY SET FOR node BY THE CALLING METHOD
void copyStackFrameToNode(metaOpenCLStackFrame * f,
		metaOpenCLStackNode ** node) {

	(*node)->frame = (*f);
	//Top-level context info
/*
	(*node)->frame.platform = f->platform;
	(*node)->frame.device = f->device;
	(*node)->frame.context = f->context;
	(*node)->frame.queue = f->queue;
//TODO Support OPENCL_SINGLE_KERNEL_PROGS
	(*node)->frame.program_opencl_core = f->program_opencl_core;

//TODO use a memcpy for the rest
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
	(*node)->frame.kernel_stencil_3d7p_db = f->kernel_stencil_3d7p_db;
	(*node)->frame.kernel_stencil_3d7p_fl = f->kernel_stencil_3d7p_fl;
	(*node)->frame.kernel_stencil_3d7p_ul = f->kernel_stencil_3d7p_ul;
	(*node)->frame.kernel_stencil_3d7p_in = f->kernel_stencil_3d7p_in;
	(*node)->frame.kernel_stencil_3d7p_ui = f->kernel_stencil_3d7p_ui;
	(*node)->frame.kernel_csr_db = f->kernel_csr_db;
	(*node)->frame.kernel_csr_fl = f->kernel_csr_fl;
	(*node)->frame.kernel_csr_ul = f->kernel_csr_ul;
	(*node)->frame.kernel_csr_in = f->kernel_csr_in;
	(*node)->frame.kernel_csr_ui = f->kernel_csr_ui;
	(*node)->frame.kernel_crc_db = f->kernel_crc_db;
	(*node)->frame.kernel_crc_fl = f->kernel_crc_fl;
	(*node)->frame.kernel_crc_ul = f->kernel_crc_ul;
	(*node)->frame.kernel_crc_in = f->kernel_crc_in;
	(*node)->frame.kernel_crc_ui = f->kernel_crc_ui;

	//Internal Buffers
	(*node)->frame.constant_face_size = f->constant_face_size;
	(*node)->frame.constant_face_stride = f->constant_face_stride;
	(*node)->frame.constant_face_child_size = f->constant_face_child_size;
	(*node)->frame.red_loc = f->red_loc;
*/
}

/// \internal For this push to be both exposed and safe from hazards, it must make a copy of the frame
/// \internal such that the user never has access to the pointer to the frame that's actually on the shared stack
/// \internal object
/// \internal This CAS-based push should be threadsafe
/// \internal WARNING assumes all parameters of metaOpenCLStackNode are set, except "next"
void metaOpenCLPushStackFrame(metaOpenCLStackFrame * frame) {
	//copy the frame, this is still the thread-private "allocated" state
	metaOpenCLStackNode * newNode = (metaOpenCLStackNode *) calloc(1,
			sizeof(metaOpenCLStackNode));
	copyStackFrameToNode(frame, &newNode);

	//grab the old node
	metaOpenCLStackNode * old = CLStack;
	//I think this is where hazards start..
	newNode->next = old;
	CLStack = newNode;
	//Only once we have successfully pushed, change the global state and trigger reinitializations
	meta_context = frame->context;
	meta_queue = frame->queue;
	meta_device = frame->device;
	meta_reinitialize_modules(module_implements_opencl);

}

metaOpenCLStackFrame * metaOpenCLTopStackFrame() {
	metaOpenCLStackFrame * frame = (metaOpenCLStackFrame *) calloc(1,
			sizeof(metaOpenCLStackFrame));
	metaOpenCLStackNode * t = CLStack; //Hazards start
	//so set a hazard pointer
	//then copy the node
	if (CLStack != NULL) {
		copyStackNodeToFrame(t, &frame);
	} else {
		free(frame);
		return NULL;
	}
	//release the hazard pointer
	//so hazards are over, return the copy and exit
	return (frame);
}

metaOpenCLStackFrame * metaOpenCLPopStackFrame() {
	metaOpenCLStackFrame * frame = (metaOpenCLStackFrame *) calloc(1,
			sizeof(metaOpenCLStackFrame));
	metaOpenCLStackNode * t = CLStack; //Hazards start
	//so set a hazard pointer
	//then copy the node
	if (CLStack != NULL) {
		copyStackNodeToFrame(t, &frame);
	} else {
		free(frame);
		return NULL;
	}
	//and pull the node off the stack
	CLStack = t->next;
	// Once we pop, we have to assume MetaMorph no longer controlls the state, so set it to the new top or NULL if we've freed the last state
	if (CLStack != NULL) {
	meta_context = CLStack->frame.context;
	meta_queue = CLStack->frame.queue;
	meta_device = CLStack->frame.device;
	meta_reinitialize_modules(module_implements_opencl);
	} else {
	meta_context = NULL;
	meta_queue = NULL;
	meta_device = NULL;
	}
	//Do something with the memory
	/// \todo FIXME: This looks like it should be on, else we're leaking free(t);
	//release the hazard pointer
	//so hazards are over, return the copy and exit
	return (frame);
}

//All this does is wrap calling metaOpenCLInitStackFrameDefault
// and setting meta_context and meta_queue appropriately
void metaOpenCLFallback() {
	metaOpenCLStackFrame * frame;
	metaOpenCLInitStackFrameDefault(&frame);
	meta_context = frame->context;
	meta_queue = frame->queue;
	meta_device = frame->device;
	metaOpenCLPushStackFrame(frame);
	free(frame); //This is safe, it's just a copy of what should now be
				 // the bottom of the stack
}

//Not meant to be called by users
__attribute__((destructor(102))) a_int meta_destroy_OpenCL() {
	//Deregister all modules that ONLY implement OpenCL
	int numOCLModules, retModCount;
	//TODO If we ever make this threadsafe, the deregister function will protect us from re-deregistration
	// but we may need to loop to make sure non get re-added between lookup and deregistration
	numOCLModules = lookup_implementing_modules(NULL, 0, module_implements_opencl, false);
	a_module_record ** oclModules = (a_module_record **)calloc(sizeof(a_module_record *), numOCLModules);
	retModCount = lookup_implementing_modules(oclModules, numOCLModules, module_implements_opencl, false);
	a_err deregErr;
	for (; retModCount > 0; retModCount--) {
		deregErr = meta_deregister_module(oclModules[retModCount-1]->module_registry_func);
	}
	//
	metaOpenCLStackFrame * frame = metaOpenCLPopStackFrame();
	while (frame != NULL) {
		metaOpenCLDestroyStackFrame(frame);
		frame = metaOpenCLPopStackFrame();
	}
	return 0; //TODO real return code
}

a_int meta_get_state_OpenCL(cl_platform_id * platform, cl_device_id * device,
		cl_context * context, cl_command_queue * queue) {
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (platform != NULL)
		*platform = frame->platform;
	if (device != NULL)
		*device = frame->device;
	if (context != NULL)
		*context = frame->context;
	if (queue != NULL)
		*queue = frame->queue;
	return 0; //TODO real return code
}

a_int meta_set_state_OpenCL(cl_platform_id platform, cl_device_id device,
		cl_context context, cl_command_queue queue) {
	if (platform == NULL || device == NULL || context == NULL
			|| queue == NULL) {
		fprintf(stderr,
				"Error: meta_set_state_OpenCL requires a full frame specification!\n");
		return -1;
	}
	//Make a frame
	metaOpenCLStackFrame * frame = (metaOpenCLStackFrame *) calloc(1, 
			sizeof(metaOpenCLStackFrame));

	//copy the info into the frame
	frame->platform = platform;
	frame->device = device;
	frame->context = context;
	clRetainContext(context);
	frame->queue = queue;
	clRetainCommandQueue(queue);
	frame->state_init = 1;

	//add the metamorph programs and kernels to it
	//Don't do this unless the user explicitly asks for it (or lazily when calling a kernel
	//metaOpenCLBuildProgram(frame);

	//add the extra buffers needed for pack/unpack
	int zero = 0;
	frame->constant_face_size = clCreateBuffer(frame->context, CL_MEM_READ_ONLY,
			sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL, NULL);
	frame->constant_face_stride = clCreateBuffer(frame->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	frame->constant_face_child_size = clCreateBuffer(frame->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	frame->red_loc = clCreateBuffer(frame->context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &zero,
			NULL);

	//push it onto the stack
	metaOpenCLPushStackFrame(frame);
	free(frame);
	return 0; //TODO real return code
}

#ifdef WITH_TIMERS
//getting a pointer to specific event  
a_err meta_get_event(char * qname, char * ename, cl_event ** e)
{
	a_err ret;
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*) malloc(sizeof(metaTimerQueueFrame));

	if(strcmp(qname,"c_D2H") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[c_D2H]), ename, &frame);
	else if(strcmp(qname,"c_H2D") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[c_H2D]), ename, &frame);
	else if(strcmp(qname,"c_D2D") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[c_D2D]), ename, &frame);
	else if(strcmp(qname,"k_csr") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[k_csr]), ename, &frame);
        else if(strcmp(qname,"k_crc") == 0)
                ret = cl_get_event_node(&(metaBuiltinQueues[k_crc]), ename, &frame);
	else
		printf("Event queue '%s' is not recognized...\n",qname);	

/*	if(frame == NULL)
		printf("event node search failed ..\n");
	else
		printf("This is 'MetaMorph C core', event '%s' retrieved succesfully\n",frame->name);
*/

/*
cl_ulong start_time;
size_t return_bytes;
clGetEventProfilingInfo(frame->event.opencl,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&start_time,&return_bytes);
printf("check profiling event is correct (MM side) %lu\n",start_time);
*/
	(*e) =((cl_event *)frame->event.event_pl);	
//	(*e) =&(frame->event.opencl);
	return(ret);
}
#endif // WITH_TIMERS

cl_int metaOpenCLInitStackFrame(metaOpenCLStackFrame ** frame, cl_int device) {
	int zero = 0;
	cl_int ret = CL_SUCCESS;
	//First, make sure we've run the one-time query to initialize the device array.
	//TODO, fix synchronization on the one-time device query.
	if (__meta_platforms_array == NULL || __meta_devices_array == NULL || (((long) __meta_platforms_array) == -1)) {
		//try to perform the scan, else wait while somebody else finishes it.
		metaOpenCLQueryDevices();
	}

	//Hack to allow choose device to set mode to OpenCL, but still pick a default
	if (device == -1)
		return metaOpenCLInitStackFrameDefault(frame);

	*frame = (metaOpenCLStackFrame *) calloc(1, sizeof(metaOpenCLStackFrame));
	//TODO use the device array to do reverse lookup to figure out which platform to attach to the frame.
	//TODO ensure the device array has been initialized by somebody (even if I have to do it) before executing this block
	//TODO implement an intelligent catch for if the device number is out of range

	//copy the chosen device from the array to the new frame
	(*frame)->device = __meta_devices_array[device];
	//reverse lookup the device's platform and add it to the frame
	clGetDeviceInfo((*frame)->device, CL_DEVICE_PLATFORM,
			sizeof(cl_platform_id), &((*frame)->platform), NULL);
	//create the context and add it to the frame
	(*frame)->context = clCreateContext(NULL, 1, &((*frame)->device), NULL,
			NULL, NULL);
	(*frame)->queue = clCreateCommandQueue((*frame)->context, (*frame)->device,
			CL_QUEUE_PROFILING_ENABLE, NULL);
	//Now must be explicitly done, kernels check this metaOpenCLBuildProgram((*frame));
	// Add this debug string if needed: -g -s\"./mm_opencl_backend.cl\"
	//Allocate any internal buffers necessary for kernel functions
	(*frame)->constant_face_size = clCreateBuffer((*frame)->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	(*frame)->constant_face_stride = clCreateBuffer((*frame)->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	(*frame)->constant_face_child_size = clCreateBuffer((*frame)->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	(*frame)->red_loc = clCreateBuffer((*frame)->context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &zero,
			NULL);
	(*frame)->state_init=1;
	return ret;
}

a_err metaOpenCLInitByID(a_int id) {
  a_err ret = CL_SUCCESS;
  metaOpenCLStackFrame * frame;
  ret = metaOpenCLInitStackFrame(&frame, (cl_int) id); //no hazards, frames are thread-private
  metaOpenCLPushStackFrame(frame);//no hazards, HPs are internally managed when copying the frame to a new stack node before pushing.
  //Now it's safe to free the frame
  // But not to destroy it, as we shouldn't release the frame members
  free(frame);
  //If users request it, a full set of contexts could be pre-initialized..
  // but we can lessen overhead by only eagerly initializing one.
  //fprintf(stderr, "OpenCL Mode not yet implemented!\n");
  return ret;
}

a_err metaOpenCLCurrDev(a_int * id) {
  //Make sure some context exists..
  if (meta_context == NULL) metaOpenCLFallback();
  *id = -1;
  int i;
  for (i= 0; i < num_devices; i++) {
    if (__meta_devices_array[i] == meta_device) *id = i;
  }
  return (*id == -1 ? -1 : CL_SUCCESS);
}

a_err metaOpenCLMaxWorkSizes(a_dim3 * work_groups, a_dim3 * work_items) {
  a_err ret = CL_SUCCESS;
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallback();
		size_t max_wg_size, max_wg_dim_sizes[3];
		ret = clGetDeviceInfo(meta_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
		ret |= clGetDeviceInfo(meta_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, &max_wg_dim_sizes, NULL);
		if ((*work_items)[0] * (*work_items)[1] * (*work_items)[2] > max_wg_size)
		{	fprintf(stderr, "Error: Maximum block volume is: %lu\nRequested block volume of: %lu (%lu * %lu * %lu) not supported!\n", max_wg_size, (*work_items)[0] * (*work_items)[1] * (*work_items)[2], (*work_items)[0], (*work_items)[1], (*work_items)[2]); ret |= -1;}

		if ((*work_items)[0] > max_wg_dim_sizes[0]) {fprintf(stderr, "Error: Maximum block size for dimension 0 is: %lu\nRequested 0th dimension size of: %lu not supported\n!", max_wg_dim_sizes[0], (*work_items)[0]); ret |= -1;}
		if ((*work_items)[1] > max_wg_dim_sizes[1]) {fprintf(stderr, "Error: Maximum block size for dimension 1 is: %lu\nRequested 1st dimension size of: %lu not supported\n!", max_wg_dim_sizes[1], (*work_items)[1]); ret |= -1;}
		if ((*work_items)[2] > max_wg_dim_sizes[2]) {fprintf(stderr, "Error: Maximum block size for dimension 2 is: %lu\nRequested 2nd dimension size of: %lu not supported\n!", max_wg_dim_sizes[2], (*work_items)[2]); ret |= -1;}
		return ret;
}

a_err metaOpenCLFlush() {
  a_err ret = CL_SUCCESS;
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallback();
		clFinish(meta_queue);
  return ret;
}

//Just a small wrapper around a cl_event allocator to keep the real datatype exclusively inside the OpenCL backend.
a_err metaOpenCLCreateEvent(void ** ret_event) {
  a_err ret = CL_SUCCESS;
  if (ret_event != NULL) *ret_event = malloc(sizeof(cl_event));
  else ret = CL_INVALID_EVENT;
  return ret;
}

//A simple wrapper to get the start time of a meta_event that contains a cl_event
a_err metaOpenCLEventStartTime(meta_event event, unsigned long * ret_time) {
  a_err ret = CL_SUCCESS;
  if (event.mode == metaModePreferOpenCL && event.event_pl != NULL && ret_time != NULL) ret = clGetEventProfilingInfo(*((cl_event *)event.event_pl), CL_PROFILING_COMMAND_START, sizeof(unsigned long), ret_time, NULL);
  else ret = CL_INVALID_EVENT;
  return ret;
}

//A simple wrapper to get the end time of a meta_event that contains a cl_event
a_err metaOpenCLEventEndTime(meta_event event, unsigned long * ret_time) {
  a_err ret = CL_SUCCESS;
  if (event.mode == metaModePreferOpenCL && event.event_pl != NULL && ret_time != NULL) ret = clGetEventProfilingInfo(*((cl_event *)event.event_pl), CL_PROFILING_COMMAND_END, sizeof(unsigned long), ret_time, NULL);
  else ret = CL_INVALID_EVENT;
  return ret;
}

//calls all the necessary CLRelease* calls for frame members
//DOES NOT:
//	pop any stack nodes
//	free any stack nodes
//	implement any thread safety, frames should always be thread private, only the stack should be shared, and all franes must be copied to or from stack nodes using the hazard-aware copy methods.
//	 (more specifically, copying a frame to a node doesn't need to be hazard-aware, as the node cannot be shared unless copied inside the hazard-aware metaOpenCLPushStackFrame. Pop, Top, and copyStackNodeToFrame are all hazard aware and provide a thread-private copy back to the caller.)
cl_int metaOpenCLDestroyStackFrame(metaOpenCLStackFrame * frame) {
	cl_int ret = CL_SUCCESS;
	//Since we always calloc the frames, if a kernel or program is uninitialized it will have a value of zero so we can simply safety check the release calls
	//Release Kernels
	if (frame->kernel_reduce_db) clReleaseKernel(frame->kernel_reduce_db);
	if (frame->kernel_reduce_fl) clReleaseKernel(frame->kernel_reduce_fl);
	if (frame->kernel_reduce_ul) clReleaseKernel(frame->kernel_reduce_ul);
	if (frame->kernel_reduce_in) clReleaseKernel(frame->kernel_reduce_in);
	if (frame->kernel_reduce_ui) clReleaseKernel(frame->kernel_reduce_ui);
	if (frame->kernel_dotProd_db) clReleaseKernel(frame->kernel_dotProd_db);
	if (frame->kernel_dotProd_fl) clReleaseKernel(frame->kernel_dotProd_fl);
	if (frame->kernel_dotProd_ul) clReleaseKernel(frame->kernel_dotProd_ul);
	if (frame->kernel_dotProd_in) clReleaseKernel(frame->kernel_dotProd_in);
	if (frame->kernel_dotProd_ui) clReleaseKernel(frame->kernel_dotProd_ui);
	if (frame->kernel_transpose_2d_face_db) clReleaseKernel(frame->kernel_transpose_2d_face_db);
	if (frame->kernel_transpose_2d_face_fl) clReleaseKernel(frame->kernel_transpose_2d_face_fl);
	if (frame->kernel_transpose_2d_face_ul) clReleaseKernel(frame->kernel_transpose_2d_face_ul);
	if (frame->kernel_transpose_2d_face_in) clReleaseKernel(frame->kernel_transpose_2d_face_in);
	if (frame->kernel_transpose_2d_face_ui) clReleaseKernel(frame->kernel_transpose_2d_face_ui);
	if (frame->kernel_pack_2d_face_db) clReleaseKernel(frame->kernel_pack_2d_face_db);
	if (frame->kernel_pack_2d_face_fl) clReleaseKernel(frame->kernel_pack_2d_face_fl);
	if (frame->kernel_pack_2d_face_ul) clReleaseKernel(frame->kernel_pack_2d_face_ul);
	if (frame->kernel_pack_2d_face_in) clReleaseKernel(frame->kernel_pack_2d_face_in);
	if (frame->kernel_pack_2d_face_ui) clReleaseKernel(frame->kernel_pack_2d_face_ui);
	if (frame->kernel_unpack_2d_face_db) clReleaseKernel(frame->kernel_unpack_2d_face_db);
	if (frame->kernel_unpack_2d_face_fl) clReleaseKernel(frame->kernel_unpack_2d_face_fl);
	if (frame->kernel_unpack_2d_face_ul) clReleaseKernel(frame->kernel_unpack_2d_face_ul);
	if (frame->kernel_unpack_2d_face_in) clReleaseKernel(frame->kernel_unpack_2d_face_in);
	if (frame->kernel_unpack_2d_face_ui) clReleaseKernel(frame->kernel_unpack_2d_face_ui);
	if (frame->kernel_stencil_3d7p_db) clReleaseKernel(frame->kernel_stencil_3d7p_db);
	if (frame->kernel_stencil_3d7p_fl) clReleaseKernel(frame->kernel_stencil_3d7p_fl);
	if (frame->kernel_stencil_3d7p_ul) clReleaseKernel(frame->kernel_stencil_3d7p_ul);
	if (frame->kernel_stencil_3d7p_in) clReleaseKernel(frame->kernel_stencil_3d7p_in);
	if (frame->kernel_stencil_3d7p_ui) clReleaseKernel(frame->kernel_stencil_3d7p_ui);
	if (frame->kernel_csr_db) clReleaseKernel(frame->kernel_csr_db);
	if (frame->kernel_csr_fl) clReleaseKernel(frame->kernel_csr_fl);
	if (frame->kernel_csr_ul) clReleaseKernel(frame->kernel_csr_ul);
	if (frame->kernel_csr_in) clReleaseKernel(frame->kernel_csr_in);
	if (frame->kernel_csr_ui) clReleaseKernel(frame->kernel_csr_ui);
//	clReleaseKernel(frame->kernel_crc_db);
//	clReleaseKernel(frame->kernel_crc_fl);
//	clReleaseKernel(frame->kernel_crc_ul);
//	clReleaseKernel(frame->kernel_crc_in);
	if (frame->kernel_crc_ui) clReleaseKernel(frame->kernel_crc_ui);

	//Release Internal Buffers
	clReleaseMemObject(frame->constant_face_size);
	clReleaseMemObject(frame->constant_face_stride);
	clReleaseMemObject(frame->constant_face_child_size);
	clReleaseMemObject(frame->red_loc);

	//Release remaining context info
#ifndef OPENCL_SINGLE_KERNEL_PROGS
	if (frame->program_opencl_core) clReleaseProgram(frame->program_opencl_core);
#else
	if (frame->program_reduce_db) clReleaseProgram(frame->program_reduce_db);
	if (frame->program_reduce_fl) clReleaseProgram(frame->program_reduce_fl);
	if (frame->program_reduce_ul) clReleaseProgram(frame->program_reduce_ul);
	if (frame->program_reduce_in) clReleaseProgram(frame->program_reduce_in);
	if (frame->program_reduce_ui) clReleaseProgram(frame->program_reduce_ui);
	if (frame->program_dotProd_db) clReleaseProgram(frame->program_dotProd_db);
	if (frame->program_dotProd_fl) clReleaseProgram(frame->program_dotProd_fl);
	if (frame->program_dotProd_ul) clReleaseProgram(frame->program_dotProd_ul);
	if (frame->program_dotProd_in) clReleaseProgram(frame->program_dotProd_in);
	if (frame->program_dotProd_ui) clReleaseProgram(frame->program_dotProd_ui);
	if (frame->program_transpose_2d_face_db) clReleaseProgram(frame->program_transpose_2d_face_db);
	if (frame->program_transpose_2d_face_fl) clReleaseProgram(frame->program_transpose_2d_face_fl);
	if (frame->program_transpose_2d_face_ul) clReleaseProgram(frame->program_transpose_2d_face_ul);
	if (frame->program_transpose_2d_face_in) clReleaseProgram(frame->program_transpose_2d_face_in);
	if (frame->program_transpose_2d_face_ui) clReleaseProgram(frame->program_transpose_2d_face_ui);
	if (frame->program_pack_2d_face_db) clReleaseProgram(frame->program_pack_2d_face_db);
	if (frame->program_pack_2d_face_fl) clReleaseProgram(frame->program_pack_2d_face_fl);
	if (frame->program_pack_2d_face_ul) clReleaseProgram(frame->program_pack_2d_face_ul);
	if (frame->program_pack_2d_face_in) clReleaseProgram(frame->program_pack_2d_face_in);
	if (frame->program_pack_2d_face_ui) clReleaseProgram(frame->program_pack_2d_face_ui);
	if (frame->program_unpack_2d_face_db) clReleaseProgram(frame->program_unpack_2d_face_db);
	if (frame->program_unpack_2d_face_fl) clReleaseProgram(frame->program_unpack_2d_face_fl);
	if (frame->program_unpack_2d_face_ul) clReleaseProgram(frame->program_unpack_2d_face_ul);
	if (frame->program_unpack_2d_face_in) clReleaseProgram(frame->program_unpack_2d_face_in);
	if (frame->program_unpack_2d_face_ui) clReleaseProgram(frame->program_unpack_2d_face_ui);
	if (frame->program_3d7p_db) clReleaseProgram(frame->program_stencil_3d7p_db);
	if (frame->program_3d7p_fl) clReleaseProgram(frame->program_stencil_3d7p_fl);
	if (frame->program_3d7p_ul) clReleaseProgram(frame->program_stencil_3d7p_ul);
	if (frame->program_3d7p_in) clReleaseProgram(frame->program_stencil_3d7p_in);
	if (frame->program_3d7p_ui) clReleaseProgram(frame->program_stencil_3d7p_ui);
	if (frame->program_csr_db) clReleaseProgram(frame->program_csr_db);
	if (frame->program_csr_fl) clReleaseProgram(frame->program_csr_fl);
	if (frame->program_csr_ul) clReleaseProgram(frame->program_csr_ul);
	if (frame->program_csr_in) clReleaseProgram(frame->program_csr_in);
	if (frame->program_csr_ui) clReleaseProgram(frame->program_csr_ui);
//	clReleaseProgram(frame->program_crc_db);
//	clReleaseProgram(frame->program_crc_fl);
//	clReleaseProgram(frame->program_crc_ul);
//	clReleaseProgram(frame->program_crc_in);
	if (frame->program_crc_ui) clReleaseProgram(frame->program_crc_ui);
	
#endif
	frame->kernels_init = 0;
	clReleaseCommandQueue(frame->queue);
	clReleaseContext(frame->context);
	frame->state_init = 0;
	//release the frame's program source
#if defined(OPENCL_SINGLE_KERNEL_PROGS) && defined(WITH_INTELFPGA)
	free((void *) frame->metaCLbin_reduce_db);
	frame->metaCLbinLen_reduce_db = 0;
	free((void *) frame->metaCLbin_reduce_fl);
	frame->metaCLbinLen_reduce_fl = 0;
	free((void *) frame->metaCLbin_reduce_ul);
	frame->metaCLbinLen_reduce_ul = 0;
	free((void *) frame->metaCLbin_reduce_in);
	frame->metaCLbinLen_reduce_in = 0;
	free((void *) frame->metaCLbin_reduce_ui);
	frame->metaCLbinLen_reduce_ui = 0;
	free((void *) frame->metaCLbin_dotProd_db);
	frame->metaCLbinLen_dotProd_db = 0;
	free((void *) frame->metaCLbin_dotProd_fl);
	frame->metaCLbinLen_dotProd_fl = 0;
	free((void *) frame->metaCLbin_dotProd_ul);
	frame->metaCLbinLen_dotProd_ul = 0;
	free((void *) frame->metaCLbin_dotProd_in);
	frame->metaCLbinLen_dotProd_in = 0;
	free((void *) frame->metaCLbin_dotProd_ui);
	frame->metaCLbinLen_dotProd_ui = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_db);
	frame->metaCLbinLen_transpose_2d_face_db = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_fl);
	frame->metaCLbinLen_transpose_2d_face_fl = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_ul);
	frame->metaCLbinLen_transpose_2d_face_ul = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_in);
	frame->metaCLbinLen_transpose_2d_face_in = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_ui);
	frame->metaCLbinLen_transpose_2d_face_ui = 0;
	free((void *) frame->metaCLbin_pack_2d_face_db);
	frame->metaCLbinLen_pack_2d_face_db = 0;
	free((void *) frame->metaCLbin_pack_2d_face_fl);
	frame->metaCLbinLen_pack_2d_face_fl = 0;
	free((void *) frame->metaCLbin_pack_2d_face_ul);
	frame->metaCLbinLen_pack_2d_face_ul = 0;
	free((void *) frame->metaCLbin_pack_2d_face_in);
	frame->metaCLbinLen_pack_2d_face_in = 0;
	free((void *) frame->metaCLbin_pack_2d_face_ui);
	frame->metaCLbinLen_pack_2d_face_ui = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_db);
	frame->metaCLbinLen_unpack_2d_face_db = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_fl);
	frame->metaCLbinLen_unpack_2d_face_fl = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_ul);
	frame->metaCLbinLen_unpack_2d_face_ul = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_in);
	frame->metaCLbinLen_unpack_2d_face_in = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_ui);
	frame->metaCLbinLen_unpack_2d_face_ui = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_db);
	frame->metaCLbinLen_stencil_3d7p_db = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_fl);
	frame->metaCLbinLen_stencil_3d7p_fl = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_ul);
	frame->metaCLbinLen_stencil_3d7p_ul = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_in);
	frame->metaCLbinLen_stencil_3d7p_in = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_ui);
	frame->metaCLbinLen_stencil_3d7p_ui = 0;
	free((void *) frame->metaCLbin_csr_db);
	frame->metaCLbinLen_csr_db = 0;
	free((void *) frame->metaCLbin_csr_fl);
	frame->metaCLbinLen_csr_fl = 0;
	free((void *) frame->metaCLbin_csr_ul);
	frame->metaCLbinLen_csr_ul = 0;
	free((void *) frame->metaCLbin_csr_in);
	frame->metaCLbinLen_csr_in = 0;
	free((void *) frame->metaCLbin_csr_ui);
	frame->metaCLbinLen_csr_ui = 0;
//	free((void *) frame->metaCLbin_crc_db);
//	frame->metaCLbinLen_crc_db = 0;
//	free((void *) frame->metaCLbin_crc_fl);
//	frame->metaCLbinLen_crc_fl = 0;
//	free((void *) frame->metaCLbin_crc_ul);
//	frame->metaCLbinLen_crc_ul = 0;
//	free((void *) frame->metaCLbin_crc_in);
//	frame->metaCLbinLen_crc_in = 0;
	free((void *) frame->metaCLbin_crc_ui);
	frame->metaCLbinLen_crc_ui = 0;
#else
	free((void *) frame->metaCLProgSrc);
	frame->metaCLProgLen = 0;
#endif
	return ret; //TODO implement real return code
}

//This is a fallback catchall to ensure some context is initialized
// iff the user calls an accelerator function in OpenCL mode before
// calling meta_set_acc in OpenCL mode.
//It will pick some valid OpenCL device, and emit a warning to stderr
// that all OpenCL calls until meta_set_acc will refer to this device
//It also implements environment-variable-controlled device selection
// via the "TARGET_DEVICE" string environemnt variable, which must match
// EXACTLY the device name reported to the OpenCL runtime.
//TODO implement a reasonable passthrough for any errors which OpenCL may throw.
cl_int metaOpenCLInitStackFrameDefault(metaOpenCLStackFrame ** frame) {
	cl_int ret = CL_SUCCESS;
	//First, make sure we've run the one-time query to initialize the device array
	if (__meta_platforms_array == NULL || __meta_devices_array == NULL || (((long) __meta_platforms_array) == -1)) {
		//try to perform the scan, else wait while somebody else finishes it.
		metaOpenCLQueryDevices();
	}

	//Simply print the names of all devices, to assist later environment-variable device selection
	fprintf(stderr, "WARNING: Automatic OpenCL device selection used!\n");
	fprintf(stderr, "\tThe following devices are identified in the system:\n");
	char buff[128];
	int i;
	for (i = 0; i < num_devices; i++) {
		clGetDeviceInfo(__meta_devices_array[i], CL_DEVICE_NAME, 128, (void *) &buff[0],
				NULL);
		fprintf(stderr, "Device [%d]: \"%s\"\n", i, buff);
	}

	//This is how you pick a specific device using an environment variable

	int gpuID = -1;

	if (getenv("TARGET_DEVICE") != NULL) {
		gpuID = metaOpenCLGetDeviceID(getenv("TARGET_DEVICE"), &__meta_devices_array[0],
				num_devices);
		if (gpuID < 0)
			fprintf(stderr,
					"Device \"%s\" not found.\nDefaulting to first device found.\n",
					getenv("TARGET_DEVICE"));
	} else {
		fprintf(stderr,
				"Environment variable TARGET_DEVICE not set.\nDefaulting to first device found.\n");
	}

	gpuID = gpuID < 0 ? 0 : gpuID; //Ternary check to make sure gpuID is valid, if it's less than zero, default to zero, otherwise keep

	clGetDeviceInfo(__meta_devices_array[gpuID], CL_DEVICE_NAME, 128, (void *) &buff[0],
			NULL);
	fprintf(stderr, "Selected Device %d: %s\n", gpuID, buff);

	//Now that we've picked a reasonable default, fill in the details for the frame object
	ret = metaOpenCLInitStackFrame(frame, gpuID);

	return (ret);
}

cl_int metaOpenCLInitCoreKernels() {
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	cl_int ret_code = metaOpenCLBuildProgram(frame);
	frame->kernels_init = 1;
	return ret_code;
}

meta_cl_device_vendor metaOpenCLDetectDevice(cl_device_id dev) {
  meta_cl_device_vendor ret = meta_cl_device_vendor_unknown;
  //First, query the type of device
  cl_device_type type;
  clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
  if (type & CL_DEVICE_TYPE_CPU) ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_cpu);
  if (type & CL_DEVICE_TYPE_GPU) ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_gpu);
  if (type & CL_DEVICE_TYPE_ACCELERATOR) ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_accel);
  if (type & CL_DEVICE_TYPE_DEFAULT) ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_default);
  
  //Then query the platform, and it's name
  size_t name_size;
  char * name;
  cl_platform_id plat;
  clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &plat, NULL);
  clGetPlatformInfo(plat, CL_PLATFORM_NAME, 0, NULL, &name_size);
  name = (char *)calloc(sizeof(char), name_size+1);
  clGetPlatformInfo(plat, CL_PLATFORM_NAME, name_size+1, name, NULL);

  //and match it to known names
  if (strcmp(name, "NVIDIA CUDA") == 0) ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_nvidia);
  if (strcmp(name, "AMD Accelerated Parallel Processing") == 0) ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_amd_appsdk);
  //TODO AMD ROCM
  //TODO Intel CPU/GPU
  if (strcmp(name, "Intel(R) FPGA SDK for OpenCL(TM)") == 0) ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_intelfpga);
  if (strcmp(name, "Intel(R) FPGA Emulation Platform for OpenCL(TM)") == 0) ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_intelfpga);
  if (strcmp(name, "Altera SDK for OpenCL") == 0) ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_intelfpga);
  //TODO Xilinx
  if (strcmp(name, "Portable Computing Language") == 0) ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_pocl);
  
  if ((ret & meta_cl_device_vendor_mask) == meta_cl_device_vendor_unknown) fprintf(stderr, "Warning: OpenCL platform vendor \"%s\" is unrecognized, assuming .cl inputs and JIT support\n", name);
  free(name);
  return ret;
}

a_err metaOpenCLAlloc(void ** ptr, size_t size) {
  a_err ret;
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallback();
		*ptr = (void *) clCreateBuffer(meta_context, CL_MEM_READ_WRITE, size, NULL, (cl_int *)&ret);
  return ret;
}

a_err metaOpenCLFree(void* ptr) {
  a_err ret;
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallback();
		ret = clReleaseMemObject((cl_mem)ptr);
  return ret;
}

a_err metaOpenCLWrite(void * dst, void * src, size_t size, a_bool async, meta_callback *call, void *call_pl, meta_event * ret_event) {
  a_err ret;
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallback();
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
		ret = clEnqueueWriteBuffer(meta_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 0, NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback)call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = size;
//	(timer_frame->event.opencl) = event;
	//by creating a timer frame, we also create a meta_event, need to initialize it
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[c_H2D]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;
  return ret;
}

a_err metaOpenCLRead(void * dst, void * src, size_t size, a_bool async, meta_callback *call, void *call_pl, meta_event * ret_event) {
  a_err ret;
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallback();
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
		ret = clEnqueueReadBuffer(meta_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback)call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = size;
//	(timer_frame->event.opencl) = event;
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[c_D2H]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;
  return ret;
}

a_err metaOpenCLDevCopy(void * dst, void * src, size_t size, a_bool async, meta_callback *call, void *call_pl, meta_event * ret_event) {
  a_err ret;
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallback();
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
		ret = clEnqueueCopyBuffer(meta_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size, 0, NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback)call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = size;
//	(timer_frame->event.opencl) = event;
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[c_D2D]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;
		//clEnqueueCopyBuffer is by default async, so clFinish
		if (!async) clFinish(meta_queue);
  return ret;
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
a_err opencl_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data1, void * data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], void * reduced_val,
		meta_type_id type, int async, meta_callback *call, void *call_pl, meta_event * ret_event) {
	a_err ret;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3];
	size_t block[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
	int iters;

	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = (((*arr_end)[0] - (*arr_start)[0] + (block[0])) / block[0])
				* block[0];
		grid[1] = (((*arr_end)[1] - (*arr_start)[1] + (block[1])) / block[1])
				* block[1];
		grid[2] = block[2];
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block[2])) / block[2]);
	} else {
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
		iters = (*grid_size)[2];
	}
	smem_len = block[0] * block[1] * block[2];
	//before enqueuing, get a copy of the top stack frame
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);

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
		fprintf(stderr,
				"Error: Function 'opencl_dotProd' not implemented for selected type!\n");
		return -1;
		break;

	}
	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &data1);
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
	ret |= clSetKernelArg(kern, 11, sizeof(cl_int), &iters);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_mem *), &reduced_val);
	ret |= clSetKernelArg(kern, 13, sizeof(cl_int), &smem_len);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_dotProd'!\n");
	}

  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0,
			NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type);
//	(timer_frame->event.opencl) = event;
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[k_dotProd]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

a_err opencl_reduce(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data, size_t (*array_size)[3], size_t (*arr_start)[3],
		size_t (*arr_end)[3], void * reduced_val, meta_type_id type, int async,
		meta_callback *call, void *call_pl, meta_event * ret_event) {
	a_err ret;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3];
	size_t block[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
	int iters;
	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = (((*arr_end)[0] - (*arr_start)[0] + (block[0])) / block[0])
				* block[0];
		grid[1] = (((*arr_end)[1] - (*arr_start)[1] + (block[1])) / block[1])
				* block[1];
		grid[2] = block[2];
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block[2])) / block[2]);
	} else {
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
		iters = (*grid_size)[2];
	}
	smem_len = block[0] * block[1] * block[2];
	//before enqueuing, get a copy of the top stack frame
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);

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
		fprintf(stderr,
				"Error: Function 'opencl_reduce' not implemented for selected type!\n");
		return -1;
		break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &data);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_int), &(*array_size)[0]);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*array_size)[1]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*array_size)[2]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*arr_start)[0]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*arr_start)[1]);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_int), &(*arr_start)[2]);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_int), &(*arr_end)[0]);
	ret |= clSetKernelArg(kern, 8, sizeof(cl_int), &(*arr_end)[1]);
	ret |= clSetKernelArg(kern, 9, sizeof(cl_int), &(*arr_end)[2]);
	ret |= clSetKernelArg(kern, 10, sizeof(cl_int), &iters);
	ret |= clSetKernelArg(kern, 11, sizeof(cl_mem *), &reduced_val);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_int), &smem_len);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_reduce'!\n");
	}
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0,
			NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type);
//	(timer_frame->event.opencl) = event;
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[k_reduce]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_transpose_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *indata, void *outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3], meta_type_id type, int async, meta_callback * call, void *call_pl, meta_event * ret_event) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3];
	size_t block[3] = { TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, 1 };
	//cl_int smem_len = (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
// TODO update to use user provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};\
	//FIXME: make this smart enough to rescale the threadblock (and thus shared memory - e.g. bank conflicts) w.r.t. double vs. float
	if (grid_size == NULL || block_size == NULL) {
		//FIXME: reconcile TILE_DIM/BLOCK_ROWS
		grid[0] = (((*tran_dim_xy)[0] + block[0] - 1) / block[0]) * block[0];
		grid[1] = (((*tran_dim_xy)[1] + block[1] - 1) / block[1]) * block[1];
		grid[2] = 1;
	} else {
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
	}
	//The +1 here is to avoid bank conflicts with 32 floats or 16 doubles and is required by the kernel logic
	smem_len = (block[0] + 1) * block[1] * block[2];
	//TODO as the frame grows larger with more kernels, this overhead will start to add up
	// Need a better (safe) way of accessing the stack for kernel launches
	//before enqueuing, get a copy of the top stack frame
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);

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
		fprintf(stderr,
				"Error: Function 'opencl_transpose_face' not implemented for selected type!\n");
		return -1;
		break;

	}
	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &outdata);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &indata);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*arr_dim_xy)[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*arr_dim_xy)[1]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*tran_dim_xy)[0]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*tran_dim_xy)[1]);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_transpose_face'!\n");
	}
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 2, NULL, grid, block, 0,
			NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = (*tran_dim_xy)[0]*(*tran_dim_xy)[1]*get_atype_size(type);
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[k_transpose_2d_face]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, meta_callback * call, void *call_pl, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3) {
	cl_int ret;
	cl_kernel kern;
	cl_int size = face->size[0] * face->size[1] * face->size[2];
	cl_int smem_size;
	size_t grid[3];
	size_t block[3] = { 256, 1, 1 };
	//before enqueuing, get a copy of the top stack frame
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);

	//copy required pieces of the face struct into constant memory
  cl_event event_c1;
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferOpenCL && ret_event_c1->event_pl != NULL) event_c1 = *((cl_event *)ret_event_c1->event_pl);
	ret = clEnqueueWriteBuffer(frame->queue, frame->constant_face_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->size, 0, NULL, &event_c1);
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferOpenCL && ret_event_c1->event_pl != NULL) *((cl_event *)ret_event_c1->event_pl) = event_c1;
  cl_event event_c2;
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferOpenCL && ret_event_c2->event_pl != NULL) event_c2 = *((cl_event *)ret_event_c2->event_pl);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_stride,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->stride, 0, NULL, &event_c2);
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferOpenCL && ret_event_c2->event_pl != NULL) *((cl_event *)ret_event_c2->event_pl) = event_c2;
  cl_event event_c3;
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferOpenCL && ret_event_c3->event_pl != NULL) event_c3 = *((cl_event *)ret_event_c3->event_pl);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_child_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			remain_dim, 0, NULL, &event_c3);
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferOpenCL && ret_event_c3->event_pl != NULL) *((cl_event *)ret_event_c3->event_pl) = event_c3;
//TODO update to use user-provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = ((size + block[0] - 1) / block[0]) * block[0];
		grid[1] = 1;
		grid[2] = 1;
	} else {
		//This is a workaround for some non-determinism that was observed when allowing fully-arbitrary spec of grid/block
		if ((*block_size)[1] != 1 || (*block_size)[2] != 1
				|| (*grid_size)[1] != 1 || (*grid_size)[2])
			fprintf(stderr,
					"WARNING: Pack requires 1D block and grid, ignoring y/z params!\n");
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = 1;
		grid[2] = 1;
		//grid[1] = (*grid_size)[1]*(*block_size)[1];
		//grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = 1;
		block[2] = 1;
		//block[1] = (*block_size)[1];
		//block[2] = (*block_size)[2];
	}
	smem_size = face->count * block[0] * sizeof(int);
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
		fprintf(stderr,
				"Error: Function 'opencl_pack_face' not implemented for selected type!\n");
		return -1;
		break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &packed_buf);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &buf);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &size);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(face->start));
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(face->count));
	ret |= clSetKernelArg(kern, 5, sizeof(cl_mem *),
			&frame->constant_face_size);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_mem *),
			&frame->constant_face_stride);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_mem *),
			&frame->constant_face_child_size);
	ret |= clSetKernelArg(kern, 8, smem_size, NULL);
  cl_event event_k1;
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferOpenCL && ret_event_k1->event_pl != NULL) event_k1 = *((cl_event *)ret_event_k1->event_pl);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, grid, block, 0,
			NULL, &event_k1);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event_k1, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame_k1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * timer_frame_c1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * timer_frame_c2 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * timer_frame_c3 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame_k1->mode = metaModePreferOpenCL;
	timer_frame_c1->mode = metaModePreferOpenCL;
	timer_frame_c2->mode = metaModePreferOpenCL;
	timer_frame_c3->mode = metaModePreferOpenCL;
	timer_frame_k1->size = get_atype_size(type)*face->size[0]*face->size[1]*face->size[2];
	timer_frame_c1->size = get_atype_size(type)*3;
	timer_frame_c2->size = get_atype_size(type)*3;
	timer_frame_c3->size = get_atype_size(type)*3;
        timer_frame_k1->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_k1->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_k1->event.event_pl) = event_k1; //copy opaque event value into the new space
        timer_frame_c1->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_c1->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_c1->event.event_pl) = event_c1; //copy opaque event value into the new space
        timer_frame_c2->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_c2->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_c2->event.event_pl) = event_c2; //copy opaque event value into the new space
        timer_frame_c3->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_c3->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_c3->event.event_pl) = event_c3; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame_k1, &(metaBuiltinQueues[k_pack_2d_face]));
	metaTimerEnqueue(timer_frame_c1, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(timer_frame_c2, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(timer_frame_c3, &(metaBuiltinQueues[c_H2Dc]));
#endif
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferOpenCL && ret_event_k1->event_pl != NULL) *((cl_event *)ret_event_k1->event_pl) = event_k1;

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, meta_callback * call, void *call_pl, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3) {

	cl_int ret;
	cl_kernel kern;
	cl_int size = face->size[0] * face->size[1] * face->size[2];
	cl_int smem_size;
	size_t grid[3];
	size_t block[3] = { 256, 1, 1 };
	//before enqueuing, get a copy of the top stack frame
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);

	//copy required pieces of the face struct into constant memory
  cl_event event_c1;
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferOpenCL && ret_event_c1->event_pl != NULL) event_c1 = *((cl_event *)ret_event_c1->event_pl);
	ret = clEnqueueWriteBuffer(frame->queue, frame->constant_face_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->size, 0, NULL, &event_c1);
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferOpenCL && ret_event_c1->event_pl != NULL) *((cl_event *)ret_event_c1->event_pl) = event_c1;
  cl_event event_c2;
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferOpenCL && ret_event_c2->event_pl != NULL) event_c2 = *((cl_event *)ret_event_c2->event_pl);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_stride,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->stride, 0, NULL, &event_c2);
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferOpenCL && ret_event_c2->event_pl != NULL) *((cl_event *)ret_event_c2->event_pl) = event_c2;
  cl_event event_c3;
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferOpenCL && ret_event_c3->event_pl != NULL) event_c3 = *((cl_event *)ret_event_c3->event_pl);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_child_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			remain_dim, 0, NULL, &event_c3);
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferOpenCL && ret_event_c3->event_pl != NULL) *((cl_event *)ret_event_c3->event_pl) = event_c3;
//TODO update to use user-provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = ((size + block[0] - 1) / block[0]) * block[0];
		grid[1] = 1;
		grid[2] = 1;
	} else {
		//This is a workaround for some non-determinism that was observed when allowing fully-arbitrary spec of grid/block
		if ((*block_size)[1] != 1 || (*block_size)[2] != 1
				|| (*grid_size)[1] != 1 || (*grid_size)[2])
			fprintf(stderr,
					"WARNING: Unpack requires 1D block and grid, ignoring y/z params!\n");
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = 1;
		grid[2] = 1;
		//grid[1] = (*grid_size)[1]*(*block_size)[1];
		//grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = 1;
		block[2] = 1;
		//block[1] = (*block_size)[1];
		//block[2] = (*block_size)[2];
	}
	smem_size = face->count * block[0] * sizeof(int);
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
		fprintf(stderr,
				"Error: Function 'opencl_unpack_face' not implemented for selected type!\n");
		return -1;
		break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &packed_buf);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &buf);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &size);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(face->start));
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(face->count));
	ret |= clSetKernelArg(kern, 5, sizeof(cl_mem *),
			&frame->constant_face_size);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_mem *),
			&frame->constant_face_stride);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_mem *),
			&frame->constant_face_child_size);
	ret |= clSetKernelArg(kern, 8, smem_size, NULL);
  cl_event event_k1;
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferOpenCL && ret_event_k1->event_pl != NULL) event_k1 = *((cl_event *)ret_event_k1->event_pl);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, grid, block, 0,
			NULL, &event_k1);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event_k1, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame_k1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * timer_frame_c1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * timer_frame_c2 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * timer_frame_c3 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame_k1->mode = metaModePreferOpenCL;
	timer_frame_c1->mode = metaModePreferOpenCL;
	timer_frame_c2->mode = metaModePreferOpenCL;
	timer_frame_c3->mode = metaModePreferOpenCL;
	timer_frame_k1->size = get_atype_size(type)*face->size[0]*face->size[1]*face->size[2];
	timer_frame_c1->size = get_atype_size(type)*3;
	timer_frame_c2->size = get_atype_size(type)*3;
	timer_frame_c3->size = get_atype_size(type)*3;
        timer_frame_k1->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_k1->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_k1->event.event_pl) = event_k1; //copy opaque event value into the new space
        timer_frame_c1->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_c1->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_c1->event.event_pl) = event_c1; //copy opaque event value into the new space
        timer_frame_c2->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_c2->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_c2->event.event_pl) = event_c2; //copy opaque event value into the new space
        timer_frame_c3->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame_c3->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame_c3->event.event_pl) = event_c3; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame_k1, &(metaBuiltinQueues[k_unpack_2d_face]));
	metaTimerEnqueue(timer_frame_c1, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(timer_frame_c2, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(timer_frame_c3, &(metaBuiltinQueues[c_H2Dc]));
#endif
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferOpenCL && ret_event_k1->event_pl != NULL) *((cl_event *)ret_event_k1->event_pl) = event_k1;

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * indata, void * outdata, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], meta_type_id type,
		int async, meta_callback * call, void *call_pl, meta_event * ret_event) {
	cl_int ret = CL_SUCCESS;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3] = { 1, 1, 1 };
	size_t block[3] = { 128, 2, 1 };
	//size_t block[3] = METAMORPH_OCL_DEFAULT_BLOCK;
	int iters;
	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		//do not subtract 1 from blocx for the case when end == start
		grid[0] = (((*arr_end)[0] - (*arr_start)[0] + (block[0])) / block[0]);
		grid[1] = (((*arr_end)[1] - (*arr_start)[1] + (block[1])) / block[1]);
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block[2])) / block[2]);
	} else {
//		grid[0] = (*grid_size)[0];
//		grid[1] = (*grid_size)[1];
//		block[0] = (*block_size)[0];
//		block[1] = (*block_size)[1];
//		block[2] = (*block_size)[2];
//		iters = (*grid_size)[2];
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
		iters = (*grid_size)[2];
	}
	//smem_len = (block[0]+2) * (block[1]+2) * block[2];
	smem_len = 0;
#if WITH_INTELFPGA
//TODO check why the FPGA backend uses non-zero smem_len
	smem_len = (block[0] + 2) * (block[1] + 2) * block[2];
#endif
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);

	switch (type) {
	case a_db:
		kern = frame->kernel_stencil_3d7p_db;
		break;

	case a_fl:
		kern = frame->kernel_stencil_3d7p_fl;
		break;

	case a_ul:
		kern = frame->kernel_stencil_3d7p_ul;
		break;

	case a_in:
		kern = frame->kernel_stencil_3d7p_in;
		break;

	case a_ui:
		kern = frame->kernel_stencil_3d7p_ui;
		break;

	default:
		fprintf(stderr,
				"Error: Function 'opencl_stencil_3d7p' not implemented for selected type!\n");
		return -1;
		break;
	}

	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &indata);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &outdata);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*array_size)[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*array_size)[1]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*array_size)[2]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*arr_start)[0]);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_int), &(*arr_start)[1]);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_int), &(*arr_start)[2]);
	ret |= clSetKernelArg(kern, 8, sizeof(cl_int), &(*arr_end)[0]);
	ret |= clSetKernelArg(kern, 9, sizeof(cl_int), &(*arr_end)[1]);
	ret |= clSetKernelArg(kern, 10, sizeof(cl_int), &(*arr_end)[2]);
	ret |= clSetKernelArg(kern, 11, sizeof(cl_int), &iters);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_int), &smem_len);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_stencil_3d7p'!\n");
	}
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0,
			NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type);
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[k_stencil_3d7p]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;

	if (!async)
		ret |= clFinish(frame->queue);
	free(frame);

	return (ret);

}

cl_int opencl_csr(size_t (*grid_size)[3], size_t (*block_size)[3], size_t global_size,
		void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc, 
		meta_type_id type, int async, meta_callback * call, void *call_pl, meta_event * ret_event) {

	cl_int ret = CL_SUCCESS;
	cl_kernel kern;
	
	size_t grid = 1;
	size_t block = METAMORPH_OCL_DEFAULT_BLOCK_1D;
	
	//TODO does this assume the OpenCL model or CUDA model for grid size?
	//FIXME it appears to use OpenCL model whereas everyone else uses CUDA
	if (grid_size == NULL || block_size == NULL) {
		//TODO fallback scaling
	} else {
		grid = (*grid_size)[0]*(*block_size)[0];
		block = (*block_size)[0];
	}
	
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);
	switch (type) {
	case a_db:
		kern = frame->kernel_csr_db;
		break;
	case a_fl:
		kern = frame->kernel_csr_fl;
		break;
	case a_ul:
		kern = frame->kernel_csr_ul;
		break;
	case a_in:
		kern = frame->kernel_csr_in;
		break;
	case a_ui:
		kern = frame->kernel_csr_ui;
		break;
	default:
		fprintf(stderr,
				"Error: Function 'opencl_csr' not implemented for selected type!\n");
		return -1;
		break;
	}

	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	ret = clSetKernelArg(kern, 0, sizeof(int), &global_size);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &csr_ap);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_mem *), &csr_aj);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_mem *), &csr_ax);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_mem *), &x_loc);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_mem *), &y_loc);
	
	//ret = clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, &grid, &block, 1, wait, event);
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, &grid, &block, 0, NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	timer_frame->size = block*get_atype_size(type);
	timer_frame->name = "CSR";
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[k_csr]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;
	if (!async)
		ret |= clFinish(frame->queue);
	free(frame);

	return (ret);

}


cl_int opencl_crc(void * dev_input, int page_size, int num_words, int numpages, void * dev_output, 
		meta_type_id type, int async, meta_callback * call, void *call_pl, meta_event * ret_event) {

	cl_int ret = CL_SUCCESS;
	cl_kernel kern;

//TODO it doesn't do anything with size since it uses a task, either enforce it or remove it	
//TODO I doubt there is any reason for non-FPGA platforms to use a task
//TODO Since it operates on binary data, having it typed is sort of nonsense	
	//Make sure some context exists..
	if (meta_context == NULL) metaOpenCLFallback();
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (frame->kernels_init != 1) metaOpenCLBuildProgram(frame);
	switch (type) {
	case a_db:
		kern = frame->kernel_crc_ui;
		break;
	case a_fl:
		kern = frame->kernel_crc_ui;
		break;
	case a_ul:
		kern = frame->kernel_crc_ui;
		break;
	case a_in:
		kern = frame->kernel_crc_ui;
		break;
	case a_ui:
		kern = frame->kernel_crc_ui;
		break;
	default:
		fprintf(stderr,
				"Error: Function 'opencl_csr' not implemented for selected type!\n");
		return -1;
		break;
	}

	//If the desired kernel is not initialized it will be equal to NULL
	if (((void*)kern) == NULL) return CL_INVALID_PROGRAM;
	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &dev_input);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_uint), &page_size);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_uint), &num_words);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_uint), &numpages);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_mem *), &dev_output);
	
	//ret = clEnqueueTask(frame->queue, kern, 1, wait, event);
  cl_event event;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) event = *((cl_event *)ret_event->event_pl);
	ret = clEnqueueTask(frame->queue, kern, 1, NULL, &event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(event, CL_COMPLETE, (openclCallback) call, call_pl);
#ifdef WITH_TIMERS
	metaTimerQueueFrame * timer_frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	timer_frame->mode = metaModePreferOpenCL;
	//FIXME: This doesn't seem like the "task size", ask Mohamed
	timer_frame->size = get_atype_size(type);
	timer_frame->name = "CRC";
        timer_frame->event.mode = metaModePreferOpenCL; //set mode
	metaOpenCLCreateEvent(&(timer_frame->event.event_pl)); //heap allocate
	*((cl_event *)timer_frame->event.event_pl) = event; //copy opaque event value into the new space
	metaTimerEnqueue(timer_frame, &(metaBuiltinQueues[k_crc]));
#endif
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenCL && ret_event->event_pl != NULL) *((cl_event *)ret_event->event_pl) = event;
	
	if (!async)
		ret |= clFinish(frame->queue);
	free(frame);

	return (ret);

}

#if defined(__OPENCLCC__) || defined(__cplusplus)
}
#endif

