/** \file
 * Implementation of MetaMorph's event-based profiling plugin
 */

#include "metamorph_profiling.h"
#include "metamorph_dynamic_symbols.h"
#include <dlfcn.h>
#include <string.h>

/** Maintain a static set of queues for all builtin operations (transfers and kernels) */
metaTimerQueue metaBuiltinQueues[queue_count];
/** Maintain a state variable that tells whether the profiling queues have been initialized for use */
a_bool __meta_timers_initialized = false;
/** Must be aware of what backends are available */
extern struct backend_handles backends;
/** Must be aware of what plugins are available, currently only leverage MPI rank information */
extern struct plugin_handles plugins;
/** Make sure we know what the core supports */
extern a_module_implements_backend core_capability;

/**
 * Struct to hold CUDA wrapper function pointers that are only relevant to profiling
 */
struct cuda_dyn_ptrs_profiling {
  /** Dynamically-loaded pointer to the function to read the elapsed time of a meta_event that contains two cudaEvent_ts */
  a_err (* metaCUDAEventElapsedTime)(float *, meta_event);
};
/** A global storage struct for profiling-specific functions from the CUDA backend, if it is loaded */
struct cuda_dyn_ptrs_profiling cuda_timing_funcs = {NULL};
/**
 * Struct to hold OpenCL wrapper function pointers that are only relevant to profiling
 */
struct opencl_dyn_ptrs_profiling {
  /** Dynamically-loaded pointer to the function to read the start time of a meta_event containing a cl_event */
  a_err (* metaOpenCLEventStartTime)(meta_event, unsigned long *);
  /** Dynamically-loaded pointer to the function to read the end time of a meta_event containing a cl_event */
  a_err (* metaOpenCLEventEndTime)(meta_event, unsigned long *);
};
/** A global storage struct for profiling-specific functions from the OpenCL backend, if it is loaded */
struct opencl_dyn_ptrs_profiling opencl_timing_funcs = {NULL};
/**
 * Struct to hold OpenMP wrapper function pointers that are only relevant to profiling
 */
struct openmp_dyn_ptrs_profiling {
  /** Dynamically-loaded pointer to the function to read the elapsed time of a meta_event that contains two openmpEvents */
  a_err (* metaOpenMPEventElapsedTime)(float *, meta_event);
};
/** A global storage struct for profiling-specific functions from the OpenMP backend, if it is loaded */
struct openmp_dyn_ptrs_profiling openmp_timing_funcs = {NULL};

/**
 * Struct to hold MPI function pointers that are only relevant to profiling
 */
struct mpi_dyn_ptrs_profiling {
  /** Dynamically-loaded pointer to the function to safely get the current process' MPI Rank to add to profiling information */
  a_err (* metaMPIRank)(int *);
};
/** A global storage struct for profiling-specific functions from the MPI plugin, if it is loaded */
struct mpi_dyn_ptrs_profiling mpi_timing_funcs = {NULL};

#ifdef DEPRECATED
//TODO Paul:Understand and condense this function, if possible, consider renaming
a_err cl_get_event_node(metaTimerQueue * queue, char * ename, metaTimerQueueFrame ** frame)
{
	printf("searching for event : %s\n",ename);
	metaTimerQueueNode * temp;
	metaTimerQueueNode * temp2;
	metaTimerQueueNode * h;
	metaTimerQueueNode * t;
	metaTimerQueueNode * prev;
	
	int flag = 1;
	while(1)
	{
		h = queue->head;
		if (queue->head != h)
			continue;

		t = queue->tail;
		temp = h->next;
		if (queue->head != h)
			continue;
		if (temp == NULL)
			return -1; //empty status
		/*if (h == t) {
			__sync_bool_compare_and_swap(&(queue->tail), t, n);
			continue;
		}*/
		//printf("head event is %s\n",temp->name);
		//printf("tail event is %s\n",t->name);
		while(flag > 0)
		{
/*
cl_ulong start_time;
size_t return_bytes;
cl_int err;
err = clGetEventProfilingInfo(temp->event.opencl,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&start_time,&return_bytes);
printf("passing profiling event '%s' (timer side) %lu and the error is %d\n",temp->name,start_time,err);
*/	
			if(temp == t) // exit if reached end of queue
			{
					//printf("last event search iteration\n");
					flag = 0;
			}
			//printf("entering event search ...\n");
			if(flag != 0)
				temp2 = temp->next;
			if(strcmp(temp->name , ename) == 0)//found the event
			{
				printf("found event : %s\n", temp->name);
				(*frame)->name = temp->name;
				(*frame)->event = temp->event;
				(*frame)->mode = temp->mode;
				(*frame)->size = temp->size;
				//printf("finished node to frame copy...\n");
/*
cl_ulong start_time;
size_t return_bytes;
clGetEventProfilingInfo(temp->event.opencl,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&start_time,&return_bytes);
printf("POI profiling event '%s' (timer side) %lu\n",temp->name,start_time);
*/				break;
			}
			else // keep looking
			{
				temp = temp2;
			}
		}
		if((*frame) == NULL)
		{
			printf("search for event : %s is causing an error ...\n", ename);
			exit(-1);
		}
		else
		{
			break;
		}
	}
	return 0; //success
}
#endif //DEPRECATED

a_err metaProfilingCreateTimer(meta_timer ** ret_timer, meta_preferred_mode mode, size_t size) {
  a_err ret = 0;
  if (ret_timer != NULL) {
    
    meta_timer * timer = (meta_timer *) calloc(1, sizeof(meta_timer));
    timer->mode = mode;
    timer->size = size;
    timer->event.mode = mode;
    meta_init_event(&(timer->event));
    *ret_timer = timer;
  }
  else ret = -1;
  return ret;
}
a_err metaProfilingEnqueueTimer(meta_timer timer, metaProfilingBuiltinQueueType type) {
// a_err metaTimerEnqueue(metaTimerQueueFrame * frame, metaTimerQueue * queue) {
	if (!__meta_timers_initialized) metaTimersInit();
	//allocate a new node - still in the thread-private allocated state
	metaTimerQueueNode * newnode = (metaTimerQueueNode*) malloc(
			sizeof(metaTimerQueueNode));
	//initialize the new node
	newnode->timer = timer;
	//printf("ENQUEUE %x\n", newnode);
	newnode->next = NULL;
	metaTimerQueue * queue = NULL;
	if (type >= 0 && type < queue_count) {
          queue = &(metaBuiltinQueues[type]);
        } else return -1;
	metaTimerQueueNode *t, *n;
	while (1) {
		t = queue->tail;
		//Set a hazard pointer for tail, and check it.
		if (queue->tail != t)
			continue;
		n = t->next;
		if (queue->tail != t)
			continue;
		if (n != NULL) {
			__sync_bool_compare_and_swap(&(queue->tail), t, n);
			continue;
		}
		if (__sync_bool_compare_and_swap(&(t->next), NULL, newnode))
			break;
	}
	__sync_bool_compare_and_swap(&(queue->tail), t, newnode);
	//printf("Event %s created ..... in queue %s\n",newnode->name,queue->name);
	return 0;
}

/**
 * Internal: Take the copy of the front frame on the queue, and then remove it from the queue
 * Do nothing else with the frame's copy, the caller should allocate and free it.
 * \param ret_timer the Address in which to record the dequeued timer
 * \param queue The queue to dequeue from
 * \return always returns zero
 * \todo FIXME implement proper error codes
 * \todo Make Hazard aware
 */
a_err metaTimerDequeue(meta_timer * ret_timer, metaTimerQueue * queue) {
	if (!__meta_timers_initialized) metaTimersInit();
	//TODO add a check to make sure the caller actually allocated the frame
	//TODO make this dequeue hazard-aware 
	int count = 0;
	metaTimerQueueNode *h, *t, *n;
	while (1) { //keep attempting til the dequeue is uncontended
		h = queue->head;
//		printf("DEQUEUE %d %x\n", count, h);
		count++;
		//Add hazard pointer for h, check if head is still h
		if (queue->head != h)
			continue;
		t = queue->tail;
		n = h->next;
		//add hazard pointer for n
		if (queue->head != h)
			continue;
		if (n == NULL)
			return -1; //empty status
		if (h == t) {
			__sync_bool_compare_and_swap(&(queue->tail), t, n);
			continue;
		}
		//Copy the node's data to the caller-allocated frame
		if (ret_timer != NULL) *ret_timer = n->timer;
		if (__sync_bool_compare_and_swap(&(queue->head), h, n))
			break;
	}
	//Need to make this a hazard-aware retire
//	free(h);
	return 0; //success
}

__attribute__((constructor(104)))  void metaTimersInit() {
  if (core_capability == module_uninitialized) meta_init();
	if (__meta_timers_initialized) return -1;
	//Each builtin queue needs one of these pairs
	// try to use the enum ints/names to minimize changes if new members
	// are added to the enum
	//Create a timer with a special sentinel type that we can copy into each of the queue sentinels
	meta_timer * sentinel;
	metaProfilingCreateTimer(&sentinel, metaModeUnset, 0);

	//Init the device-to-host copy queue
	metaBuiltinQueues[c_D2H].head = metaBuiltinQueues[c_D2H].tail =
			(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_D2H].head->timer = *sentinel;
	metaBuiltinQueues[c_D2H].head->next = NULL;
	metaBuiltinQueues[c_D2H].name = "Device to Host transfer";

	//Init the host-to-device copy queue
	metaBuiltinQueues[c_H2D].head = metaBuiltinQueues[c_H2D].tail =
			(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));

	metaBuiltinQueues[c_H2D].head->timer = *sentinel;
	metaBuiltinQueues[c_H2D].head->next = NULL;
	metaBuiltinQueues[c_H2D].name = "Host to Device transfer";

	//Init the host-to-host copy queue
	metaBuiltinQueues[c_H2H].head = metaBuiltinQueues[c_H2H].tail =
			(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_H2H].head->timer = *sentinel;
	metaBuiltinQueues[c_H2H].head->next = NULL;
	metaBuiltinQueues[c_H2H].name = "Host to Host transfer";

	//Init the device-to-device copy queue
	metaBuiltinQueues[c_D2D].head = metaBuiltinQueues[c_D2D].tail =
			(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_D2D].head->timer = *sentinel;
	metaBuiltinQueues[c_D2D].head->next = NULL;
	metaBuiltinQueues[c_D2D].name = "Device to Device transfer";

	//Init the host-to-device (constant) copy queue
	metaBuiltinQueues[c_H2Dc].head = metaBuiltinQueues[c_H2Dc].tail =
			(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_H2Dc].head->timer = *sentinel;
	metaBuiltinQueues[c_H2Dc].head->next = NULL;
	metaBuiltinQueues[c_H2Dc].name = "Host to Constant transfer";

	//Init the Reduction kernel queue
	metaBuiltinQueues[k_reduce].head = metaBuiltinQueues[k_reduce].tail =
			(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_reduce].head->timer = *sentinel;
	metaBuiltinQueues[k_reduce].head->next = NULL;
	metaBuiltinQueues[k_reduce].name = "Reduction Sum kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_dotProd].head = metaBuiltinQueues[k_dotProd].tail =
			(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_dotProd].head->timer = *sentinel;
	metaBuiltinQueues[k_dotProd].head->next = NULL;
	metaBuiltinQueues[k_dotProd].name = "Dot Product kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_transpose_2d_face].head =
			metaBuiltinQueues[k_transpose_2d_face].tail =
					(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_transpose_2d_face].head->timer = *sentinel;
	metaBuiltinQueues[k_transpose_2d_face].head->next = NULL;
	metaBuiltinQueues[k_transpose_2d_face].name =
			"Transpose 2DFace kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_pack_2d_face].head =
			metaBuiltinQueues[k_pack_2d_face].tail =
					(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_pack_2d_face].head->timer = *sentinel;
	metaBuiltinQueues[k_pack_2d_face].head->next = NULL;
	metaBuiltinQueues[k_pack_2d_face].name = "Pack 2DFace kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_unpack_2d_face].head =
			metaBuiltinQueues[k_unpack_2d_face].tail =
					(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_unpack_2d_face].head->timer = *sentinel;
	metaBuiltinQueues[k_unpack_2d_face].head->next = NULL;
	metaBuiltinQueues[k_unpack_2d_face].name = "Unpack 2DFace kernel call";

	//Init the stencil_3d7p kernel queue
	metaBuiltinQueues[k_stencil_3d7p].head =
			metaBuiltinQueues[k_stencil_3d7p].tail =
					(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_stencil_3d7p].head->timer = *sentinel;
	metaBuiltinQueues[k_stencil_3d7p].head->next = NULL;
	metaBuiltinQueues[k_stencil_3d7p].name = "stencil_3d7p kernel call";
	
	//Init the csr kernel queue
	metaBuiltinQueues[k_csr].head =
			metaBuiltinQueues[k_csr].tail =
					(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_csr].head->timer = *sentinel;
	metaBuiltinQueues[k_csr].head->next = NULL;
	metaBuiltinQueues[k_csr].name = "csr kernel call";
	
	//Init the csr kernel queue
	metaBuiltinQueues[k_crc].head =
			metaBuiltinQueues[k_crc].tail =
					(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_crc].head->timer = *sentinel;
	metaBuiltinQueues[k_crc].head->next = NULL;
	metaBuiltinQueues[k_crc].name = "crc kernel call";
  if (backends.cuda_be_handle != NULL) {
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDAEventElapsedTime", cuda_timing_funcs.metaCUDAEventElapsedTime);
  }
  if (backends.opencl_be_handle != NULL) {
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLEventStartTime", opencl_timing_funcs.metaOpenCLEventStartTime);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLEventEndTime", opencl_timing_funcs.metaOpenCLEventEndTime);
  }
  if (backends.openmp_be_handle != NULL) {
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPEventElapsedTime", openmp_timing_funcs.metaOpenMPEventElapsedTime);
  }
  if (plugins.mpi_handle != NULL) {
    CHECKED_DLSYM("libmm_mpi.so", plugins.mpi_handle, "metaMPIRank", mpi_timing_funcs.metaMPIRank);
  }
	__meta_timers_initialized = true;
}

/**
 * Workhorse that loops over a queue until it receives an empty signal
 * Performs work according to what METAMORPH_TIMER_LEVEL is passed in.
 * \param queue the queue to flush all timers from
 * \param level The verbosity level to report (0 = silent, 1 = aggregate, 2 = individual cals, 3 = unimplemented)
 * \todo figure out how to handle encountered events which have not completed (do we put them back on the queue? register a callback? force the command_queue to finish?
 * \todo FIXME needs to handle bad return codes
 */
void flushWorker(metaTimerQueue * queue, int level) {
	a_err ret;
	meta_timer * timer = (meta_timer *) malloc(sizeof(meta_timer));
	int val;
	unsigned long start, end, count = 0;
	size_t size = 0;
	float time = 0.0f, temp_t = 0.0f;
	while ((val = metaTimerDequeue(timer, queue)) != -1) {
		//use one loop to do everything
//		printf("JUST CHECKING %d\n", val);

		//FIXME why does only OpenCL have level 0 implementation?
		//Zero should just be eating nodes
		if(level == 0)
		{
			if (timer->mode == metaModePreferOpenCL)
			{
				
		if (opencl_timing_funcs.metaOpenCLEventStartTime != NULL) ret = (*(opencl_timing_funcs.metaOpenCLEventStartTime))(timer->event, &start);
		if (opencl_timing_funcs.metaOpenCLEventEndTime != NULL) ret = (*(opencl_timing_funcs.metaOpenCLEventEndTime))(timer->event, &end);
				temp_t = (end-start)*0.000001;
			}
		}

		if (level >= 1) {
			if (timer->mode == metaModePreferGeneric) {
				//TODO add some generic stuff
			}
			else if (timer->mode == metaModePreferCUDA) {
				//TODO add a check to cudaEventQuery to make sure frame->event.cuda[1] is finished (PS: This should be done in the CUDA backend and an appropriate backend-agnostic error returned if it's not ready)
				if(cuda_timing_funcs.metaCUDAEventElapsedTime != NULL) ret = (*(cuda_timing_funcs.metaCUDAEventElapsedTime))(&temp_t, timer->event);
			}
			else if (timer->mode == metaModePreferOpenCL) {
				//TODO add a check via clGetEventInfo to make sure the event has completed
		if (opencl_timing_funcs.metaOpenCLEventStartTime != NULL) ret = (*(opencl_timing_funcs.metaOpenCLEventStartTime))(timer->event, &start);
		if (opencl_timing_funcs.metaOpenCLEventEndTime != NULL) ret = (*(opencl_timing_funcs.metaOpenCLEventEndTime))(timer->event, &end);
				temp_t = (end-start)*0.000001;

			}
			else if (timer->mode == metaModePreferOpenMP) {
		if (openmp_timing_funcs.metaOpenMPEventElapsedTime != NULL) ret = (*(openmp_timing_funcs.metaOpenMPEventElapsedTime))(&temp_t, timer->event);
			}
			//Aggregate times/bandwidth across all 
		}
		if (level >= 2) {
			//Individual call times/bandwidths
			//TODO come up with a reasonable, generic bandwidth calculation.
			fprintf(stderr,"\t");
			int rank = -1;
			if (mpi_timing_funcs.metaMPIRank != NULL) {
				(*(mpi_timing_funcs.metaMPIRank))(&rank);
				if (rank != -1) fprintf(stderr, "Rank[%d]: ", rank);
			}
			fprintf(stderr,
					"%s [%lu] on [%lu]bytes took [%f]ms, with [%f]MB/s approximate bandwidth.\n",
					queue->name, count, timer->size, temp_t,
					(timer->size > 0 && temp_t > 0) ?
							timer->size * .001 / temp_t : 0.0);
		}

		if (level == 3) {
			//Really verbose stuff, like block/grid size
		}
		time += temp_t;
		size += timer->size;
		count++;
		//Eating the node for level 0 is inherent in the while
	}
	if (level > 0 && time > 0) {
		int rank = -1;
		if (mpi_timing_funcs.metaMPIRank != NULL) {
			(*(mpi_timing_funcs.metaMPIRank))(&rank);
			if (rank != -1) fprintf(stderr, "Rank[%d]: ", rank);
		}
		fprintf(stderr,
				"All %ss took [%f]ms, with [%f]MB/s approximate average bandwidth.\n",
				queue->name, time,
				(size > 0 && time > 0) ? size * .001 / time : 0.0);
	}
printf("Profiling event time for %s = %f\n",queue->name, time);
}

a_err metaTimersFlush() {
	//Basically, just run through all the builtin queues,
	// dequeuing each element and tallying it up
	//This is where we need METAMORPH_TIMER_LEVEL
	int i;
	char * level = NULL;
	//if  ((level = (char*)((long int) getenv("METAMORPH_TIMER_LEVEL"))) != NULL) {
	if ((level = getenv("METAMORPH_TIMER_LEVEL")) != NULL) {
		if (strcmp(level, "0") == 0) {
			for (i = 0; i < queue_count; i++)
				flushWorker(&metaBuiltinQueues[i], 0);
			//Just eat the nodes, don't do anything with them
		}

		else if (strcmp(level, "1") == 0) {
			fprintf(stderr, "***TIMER LEVEL 1 FLUSH START***\n");
			for (i = 0; i < queue_count; i++)
				flushWorker(&metaBuiltinQueues[i], 1);
			fprintf(stderr, "***TIMER LEVEL 1 FLUSH END***\n");
			//Aggregate just averages
		}

		else if (strcmp(level, "2") == 0) {
			fprintf(stderr, "***TIMER LEVEL 2 FLUSH START***\n");
			for (i = 0; i < queue_count; i++)
				flushWorker(&metaBuiltinQueues[i], 2);
			fprintf(stderr, "***TIMER LEVEL 2 FLUSH END***\n");
			//Display all individual nodes, with bandwidth (for transfers)
		}

		else if (strcmp(level, "3") == 0) {
			for (i = 0; i < queue_count; i++)
				flushWorker(&metaBuiltinQueues[i], 3);
			//Display full diagnostics
			// This should include grid and block sizes
			// offsets, device and mode, and once-only dumps at the
			// beginning (like system specs)
		}

		else {
			fprintf(stderr,
					"Error: METAMORPH_TIMER_LEVEL=\"%s\" not supported!\n",
					getenv("METAMORPH_TIMER_LEVEL"));
		}
	} else {
		for (i = 0; i < queue_count; i++)
			flushWorker(&metaBuiltinQueues[i], 0);
		//no mode is specified, just silently eat the nodes.
	}
	return 0;
}

a_err metaTimersFinish() {

	//first, make sure everything is flushed.
	if (__meta_timers_initialized) metaTimersFlush();
	__meta_timers_initialized = false;
	//then remove all reference points to these timers
	// (so that another thread can potentially spin up a separate new set..)
	//TODO timer cleanup
	return 0;
}

//TODO expose a way for users to generate their own timer queues
// Will likely require overloaded function headers for each call which
// take a queue list/count struct..

void meta_timers_init_c_() { metaTimersInit();}
void meta_timers_flush_c_() {metaTimersFlush();}
void meta_timers_finish_c_() {metaTimersFinish();}
