/*
 * The workhorse pass-through of the library. This file implements
 *  control over which modes are compiled in, and generic wrappers
 *  for all functions which must be passed through to a specific
 *  mode's backend "core" implementation.
 *
 * Should support targetting a specific class of device, a
 *  specific accelerator model (CUDA/OpenCL/OpenMP/...), and
 *  "choose best", which would be similar to Tom's CoreTSAR.
 *
 * For now, Generic, CUDA, and OpenCL are supported.
 * Generic simply uses environment variable "AFOSR_MODE" to select
 *  a mode at runtime.
 * OpenCL mode also supports selecting the "-1st" device, which
 *  forces it to refer to environment variable "TARGET_DEVICE" to attempt
 *  to string match the name of an OpenCL device, defaulting to the zeroth
 *  if no match is identified.
 *
 * TODO For OpenCL context switching, should we implement checks to ensure cl_mem
 *  regions are created w.r.t. the current context? Should it throw an error,
 *  or attempt to go through the stack and execute the call w.r.t. the correct
 *  context for the cl_mem pointer? What if a kernel launch mixes cl_mems from
 *  multiple contexts? TODO
 */

#include "afosr_cfd.h"


accel_preferred_mode run_mode = accelModePreferGeneric;

#ifdef WITH_OPENCL
cl_context accel_context = NULL;
cl_command_queue accel_queue = NULL;
cl_device_id accel_device = NULL;

//All this does is wrap calling accelOpenCLInitStackFrameDefault
// and setting accel_context and accel_queue appropriately
void accelOpenCLFallBack() {
	accelOpenCLStackFrame * frame;
	accelOpenCLInitStackFrameDefault(&frame);
	accel_context = frame->context;
	accel_queue = frame->queue;
	accel_device = frame->device;
	accelOpenCLPushStackFrame(frame);
	free(frame); //This is safe, it's just a copy of what should now be the bottom of the stack
}
#endif

//TODO - Validate attempted allocation sizes against maximum supported by the device
// particularly for OpenCL (AMD 7970 doesn't support 512 512 512 reduce size
a_err accel_alloc(void ** ptr, size_t size) {
	a_err ret;
	switch(run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement generic (runtime choice) allocation
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			ret = cudaMalloc(ptr, size);
			break;
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
			//Make sure some context exists..
			if (accel_context == NULL) accelOpenCLFallBack();
			*ptr = (void *) clCreateBuffer(accel_context, CL_MEM_READ_WRITE, size, NULL, (cl_int *)&ret);
			break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement OpenMP allocation
			break;
		#endif
	}
	return (ret);
}

//TODO implement a way for this to trigger destroying an OpenCL stack frame
// iff all cl_mems in the frame's context, as well as the frame members themselves
// have been released.
a_err accel_free(void * ptr) {
	a_err ret;
	switch (run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement generic (runtime choice) free
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			ret = cudaFree(ptr);
			break;
		#endif

		#ifdef WTIH_OPENCL
		case accelModePreferOpenCL:
			//Make sure some context exists..
			if (accel_context == NULL) accelOpenCLFallBack();
			ret = clReleaseMemObject((cl_mem)ptr);
			break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement OpenMP free
			break;
		#endif
	}
	return (ret);
}

//Simplified wrapper for sync copies
//a_err accel_copy_h2d(void * dst, void * src, size_t size) {
//	return accel_copy_h2d(dst, src, size, true);
//}
//Workhorse for both sync and async variants
a_err accel_copy_h2d(void * dst, void * src, size_t size, a_bool async) {
	a_err ret;
	#ifdef WITH_TIMERS
	accelTimerQueueFrame * frame = malloc (sizeof(accelTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	#endif
	switch (run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement generic (runtime choice) H2D copy
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[0]));
			cudaEventRecord(frame->event.cuda[0], 0);
			#endif
			if (async) {
				ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
			} else {
				ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
			}
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[1]));
			cudaEventRecord(frame->event.cuda[1], 0);
			#endif
			break;
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
			//Make sure some context exists..
			if (accel_context == NULL) accelOpenCLFallBack();
			#ifdef WITH_TIMERS
			ret = clEnqueueWriteBuffer(accel_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 0, NULL, &(frame->event.opencl));
			#else
			ret = clEnqueueWriteBuffer(accel_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 0, NULL, NULL);
			#endif
			break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement OpenMP copy
			break;
		#endif
	}
	#ifdef WITH_TIMERS
	accelTimerEnqueue(frame, &(accelBuiltinQueues[c_H2D]));
	#endif
	return (ret);
}

//Simplified wrapper for sync copies
//a_err accel_copy_d2h(void * dst, void * src, size_t) {
//	return accel_copy_d2h(dst, src, size, true);
//}
//Workhorse for both sync and async copies
a_err accel_copy_d2h(void * dst, void * src, size_t size, a_bool async) {
	a_err ret;
	#ifdef WITH_TIMERS
	accelTimerQueueFrame * frame = malloc (sizeof(accelTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	#endif
	switch (run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement generic (runtime choice) H2D copy
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[0]));
			cudaEventRecord(frame->event.cuda[0], 0);
			#endif
			if (async) {
				ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, 0);
			} else {
				ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
			}
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[1]));
			cudaEventRecord(frame->event.cuda[1], 0);
			#endif
			break;
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
			//Make sure some context exists..
			if (accel_context == NULL) accelOpenCLFallBack();
			#ifdef WITH_TIMERS
			ret = clEnqueueReadBuffer(accel_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, &(frame->event.opencl));
			#else
			ret = clEnqueueReadBuffer(accel_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, NULL);
			#endif
			break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement OpenMP copy
			break;
		#endif
	}
	#ifdef WITH_TIMERS
	accelTimerEnqueue(frame, &(accelBuiltinQueues[c_D2H]));
	#endif
	return (ret);
}

//Simplified wrapper for sync copies
//a_err accel_copy_d2d(void * dst, void * src, size_t size) {
//	return (dst, src, size, true);
//}
//Workhorse for both sync and async copies
a_err accel_copy_d2d(void * dst, void * src, size_t size, a_bool async) {
	a_err ret;
	#ifdef WITH_TIMERS
	accelTimerQueueFrame * frame = malloc (sizeof(accelTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	#endif
	switch (run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement generic (runtime choice) H2D copy
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[0]));
			cudaEventRecord(frame->event.cuda[0], 0);
			#endif
			if (async) {
				ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, 0);
			} else {
				ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
			}
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[1]));
			cudaEventRecord(frame->event.cuda[1], 0);
			#endif
			break;
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
			//Make sure some context exists..
			if (accel_context == NULL) accelOpenCLFallBack();
			#ifdef WITH_TIMERS
			ret = clEnqueueCopyBuffer(accel_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size, 0, NULL, &(frame->event.opencl));
			#else
			ret = clEnqueueCopyBuffer(accel_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size, 0, NULL, NULL);
			#endif
			//clEnqueueCopyBuffer is by default async, so clFinish
			if (!async) clFinish(accel_queue);
			break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement OpenMP copy
			break;
		#endif
	}
	#ifdef WITH_TIMERS
	accelTimerEnqueue(frame, &(accelBuiltinQueues[c_D2D]));
	#endif
	return (ret);
}

//In CUDA choosing an accelerator is optional, make sure the OpenCL version
// implements some reasonable default iff choose_accel is not called before the
// first OpenCL command
//TODO make this compatible with OpenCL initialization
//TODO unpack OpenCL platform from the uint's high short, and the device from the low short
a_err choose_accel(int accel, accel_preferred_mode mode) {
	a_err ret;
	run_mode = mode;
	switch(run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO support generic (auto-config) runtime selection
			//TODO support "AFOSR_MODE" environment variable.
			if (getenv("AFOSR_MODE") != NULL) {
				#ifdef WITH_CUDA
				if (strcmp(getenv("AFOSR_MODE"), "CUDA") == 0) return choose_accel(accel, accelModePreferCUDA);
				#endif

				#ifdef WITH_OPENCL
				if (strcmp(getenv("AFOSR_MODE"), "OpenCL") == 0 || strcmp(getenv("AFOSR_MODE"), "OpenCL_DEBUG") == 0) return choose_accel(accel, accelModePreferOpenCL);
				#endif

				#ifdef WITH_OPENMP
				if (strcmp(getenv("AFOSR_MODE"), "OpenMP") == 0) return choose_accel(accel, accelModePreferOpenMP);
				#endif

				#ifdef WITH_CORETSAR
				if (strcmp(getenv("AFOSR_MODE"), "CoreTsar") == 0) {
					fprintf(stderr, "CoreTsar mode not yet supported!\n");
					//TODO implement whatever's required to get CoreTsar going...

				}
				#endif

				fprintf(stderr, "Error: AFOSR_MODE=\"%s\" not supported with specified compiler definitions.\n", getenv("AFOSR_MODE"));
			}
			fprintf(stderr, "Generic Mode only supported with \"AFOSR_MODE\" environment variable set to one of:\n");
			#ifdef WITH_CUDA
			fprintf(stderr, "\"CUDA\"\n");
			#endif
			#ifdef WITH_OPENCL
			fprintf(stderr, "\"OpenCL\" (or \"OpenCL_DEBUG\")\n");
			#endif
			#ifdef WITH_OPENMP
			fprintf(stderr, "\"OpenMP\"\n");
			#endif
			#ifdef WITH_CORETSAR
			//TODO - Left as a stub to remind us that Generic Mode was intended to be used for
			// CoreTsar auto configuration across devices/modes
			fprintf(stderr, "\"CoreTsar\"\n");
			#endif
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			ret = cudaSetDevice(accel);
			printf("CUDA Mode selected with device: %d\n", accel);
			//TODO add a statement printing the device's name
			break;
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
			{accelOpenCLStackFrame * frame;
				accelOpenCLInitStackFrame(&frame, (cl_int) accel); //no hazards, frames are thread-private
				//make sure this library knows what the opencl library is using internally..
				accel_context = frame->context;
				accel_queue = frame->queue;
				accel_device = frame->device;

				accelOpenCLPushStackFrame(frame); //no hazards, HPs are internally managed when copying the frame to a new stack node before pushing.

				//Now it's safe to free the frame
				// But not to destroy it, as we shouldn't release the frame members
				free(frame);
				//If users request it, a full set of contexts could be pre-initialized..
				// but we can lessen overhead by only eagerly initializing one.
				//fprintf(stderr, "OpenCL Mode not yet implemented!\n");
			}
			break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement if needed
			fprintf(stderr, "OpenMP Mode not yet implemented!\n");
			break;
		#endif
	}
	return(ret);
}

//TODO make this compatible with OpenCL device querying
//TODO pack OpenCL platform into the uint's high short, and the device into the low short
a_err get_accel(int * accel, accel_preferred_mode * mode) {
	a_err ret;
	switch(run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement a generic response for which device was runtime selected
			fprintf(stderr, "Generic Device Query not yet implemented!\n");
			*mode = accelModePreferGeneric;
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			ret = cudaGetDevice(accel);
			*mode = accelModePreferCUDA;
			break;
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
			//Make sure some context exists..
			if (accel_context == NULL) accelOpenCLFallBack();
			//TODO implement appropriate response for OpenCL device number
			//TODO implement this based on accelOpenCLTopStackFrame
			fprintf(stderr, "OpenCL Device Query not yet implemented!\n");
			*mode = accelModePreferOpenCL;
			break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement appropriate response to indicate OpenMP is being used
			fprintf(stderr, "OpenMP Device Query not yet implemented!\n");
			*mode = accelModePreferOpenMP;
			break;
		#endif
	}
	return(ret);
}

//Stub: checks that grid_size is a valid multiple of block_size, and that
// both obey the bounds specified for the device/execution model currently in use.
// For now CUDA and OpenMP are NOOP
//For now it does not directly modify the work size at all
// nor does it return the ideal worksize in any way
// it just produces a STDERR comment stating which variable is out of bounds
// what the bound is, and what the variable is currently.
a_err accel_validate_worksize(a_dim3 * grid_size, a_dim3 * block_size) {
	a_err ret;
	switch(run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO the only reason we should still be here is CoreTsar
			// Don't worry about doing anything until when/if we add that
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA:
			//TODO implement whatever bounds checking is needed by CUDA
			return 0;
			break;
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
			//Make sure some context exists..
			if (accel_context == NULL) accelOpenCLFallBack();
			size_t max_wg_size, max_wg_dim_sizes[3];
			ret = clGetDeviceInfo(accel_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
			ret |= clGetDeviceInfo(accel_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, &max_wg_dim_sizes, NULL);
			if ((*block_size)[0] * (*block_size)[1] * (*block_size)[2] > max_wg_size)
				{fprintf(stderr, "Error: Maximum block volume is: %d\nRequested block volume of: %d (%d * %d * %d) not supported!\n", max_wg_size, (*block_size)[0] * (*block_size)[1] * (*block_size)[2], (*block_size)[0], (*block_size)[1], (*block_size)[2]); ret |= -1;}
			
			if ((*block_size)[0] > max_wg_dim_sizes[0]) {fprintf(stderr, "Error: Maximum block size for dimension 0 is: %d\nRequested 0th dimension size of: %d not supported\n!", max_wg_dim_sizes[0], (*block_size)[0]); ret |= -1;}
			if ((*block_size)[1] > max_wg_dim_sizes[1]) {fprintf(stderr, "Error: Maximum block size for dimension 1 is: %d\nRequested 1st dimension size of: %d not supported\n!", max_wg_dim_sizes[1], (*block_size)[1]); ret |= -1;}
			if ((*block_size)[2] > max_wg_dim_sizes[2]) {fprintf(stderr, "Error: Maximum block size for dimension 2 is: %d\nRequested 2nd dimension size of: %d not supported\n!", max_wg_dim_sizes[2], (*block_size)[2]); ret |= -1;}
			return(ret);
		break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
			//TODO implement any bounds checking OpenMP may need
			return 0;
			break;
		#endif
	}
	return(ret);
}

//Simple wrapper for synchronous kernel
//a_err accel_dotProd(a_dim3 * grid_size, a_dim3 * block_size, a_double * data1, a_double * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var) {
//	return accel_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, true);
//}
//Workhorse for both sync and async dot products
a_err accel_dotProd(a_dim3 * grid_size, a_dim3 * block_size, a_double * data1, a_double * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var, a_bool async) {
	a_err ret;
	#ifdef WITH_TIMERS
	accelTimerQueueFrame * frame = malloc (sizeof(accelTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*sizeof(double);
	#endif
	switch(run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement a generic reduce
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA: {
						#ifdef WITH_TIMERS
						  ret = (a_err) cuda_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, async, &(frame->event.cuda));
						#else
						  ret = (a_err) cuda_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, async, NULL);
						#endif
						  break;
					  }
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
					  //Make sure some context exists..
					  if (accel_context == NULL) accelOpenCLFallBack();
					  #ifdef WITH_TIMERS
					  ret = (a_err) opencl_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, async, &(frame->event.opencl));
					  #else
					  ret = (a_err) opencl_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, async, NULL);
					  #endif
					  break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
					  //TODO implement OpenMP reduce
					  break;
		#endif
	}
	#ifdef WITH_TIMERS
	accelTimerEnqueue(frame, &(accelBuiltinQueues[k_dotProd]));
	#endif
	return(ret);
}

//Simplified wrapper for synchronous kernel
//a_err accel_reduce(a_dim3 * grid_size, a_dim3 * block_size, a_double * data, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var) {
//	return accel_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, true);
//}
//Workhorse for both sync and async reductions
a_err accel_reduce(a_dim3 * grid_size, a_dim3 * block_size, a_double * data, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var, a_bool async) {
	a_err ret;
	#ifdef WITH_TIMERS
	accelTimerQueueFrame * frame = malloc (sizeof(accelTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*sizeof(double);
	#endif
	switch(run_mode) {
		default:
		case accelModePreferGeneric:
			//TODO implement a generic reduce
			break;

		#ifdef WITH_CUDA
		case accelModePreferCUDA: {
						#ifdef WITH_TIMERS
						  ret = (a_err) cuda_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, async, &(frame->event.cuda));
						#else
						  ret = (a_err) cuda_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, async, NULL);
						#endif
						  break;
					  }
		#endif

		#ifdef WITH_OPENCL
		case accelModePreferOpenCL:
					  //Make sure some context exists..
					  if (accel_context == NULL) accelOpenCLFallBack();
					  #ifdef WITH_TIMERS
					  ret = (a_err) opencl_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, async, &(frame->event.opencl));
					  #else
					  ret = (a_err) opencl_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, async, NULL);
					  #endif
					  break;
		#endif

		#ifdef WITH_OPENMP
		case accelModePreferOpenMP:
					  //TODO implement OpenMP reduce
					  break;
		#endif
	}
	#ifdef WITH_TIMERS
	accelTimerEnqueue(frame, &(accelBuiltinQueues[k_reduce]));
	#endif
	return(ret);
}
