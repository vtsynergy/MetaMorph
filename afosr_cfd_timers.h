/*
 * Exclusively for the core mechanics of timer imlementationts
 * None of this file should be active at all if WITH_TIMERS is not defined
 * i.e. it should only be #included in a conditional-compile block
 */
#ifndef AFOSR_CFD_TIMERS_H
#define AFOSR_CFD_TIMERS_H

//Include afosr_cfd.h to grab the properly-defined enum for modes
#ifndef AFOSR_CFD_H
	#include "afosr_cfd.h"
#endif

//Any special concerns for 
#ifdef WITH_CUDA
#endif

#ifdef WITH_OPENCL
#endif

#ifdef WITH_OPENMP
#endif

typedef union accelTimerEvent {
	#ifdef WITH_CUDA
	cudaEvent_t cuda[2];
	#endif WITH_CUDA
	#ifdef WITH_OPENCL
	cl_event opencl;
	#endif
	#ifdef WITH_OPENMP
	struct timeval openmp;
	#endif
} accelTimerEvent;

typedef struct accelTimerQueueNode {
//CUDA needs 2 events to use cudaEventElapsedTime
//OpenCL only needs one, if using the event returned from an API call
//OpenMP will either need 1 or 2, depending on if we keep start/end or just elapsed
accelTimerEvent event;
size_t size[6];
//Hijack accel_preferred_mode enum to advise the user of the frame/node how to interpret event
accel_preferred_mode mode; 
struct accelTimerQueueNode * next;
} accelTimerQueueNode;

typedef struct accelTimerQueueFrame {
//CUDA needs 2 events to use cudaEventElapsedTime
//OpenCL only needs one, if using the event returned from an API call
//OpenMP will either need 1 or 2, depending on if we keep start/end or just elapsed
accelTimerEvent event;
size_t size[6];
//Hijack accel_preferred_mode enum to advise the user of the frame/node how to interpret event
accel_preferred_mode mode; 
} accelTimerQueueFrame;

typedef struct accelTimerQueue {
	const char * name;
	accelTimerQueueNode * head, tail;
}accelTimerQueue;

//A convenience structure to tie logically-named queues to possibly-changing indicies in the Queue array
// This way we can add a new type of timer (for a kernel or copy or whatever) and potentially change indicies without needing to change a bunch of hardcoded constants.
typedef enum accelTimerQueueEnum {
	c_D2H,
	c_H2D,
	c_H2H,
	c_D2D,
	k_reduce,
	k_dotProd,
	//Special value used just to determine the size of the enum automagically
	// only works if we don't set explicit values for anything, and let it start from 0
	queue_count
};

a_err accelTimerEnqueue(accelTimerQueueFrame * frame, accelTimerQueue * queue);
a_err accelTimerDequeue(accelTimerQueueFrame ** frame, accelTimerQueue * queue);

#endif //AFOSR_CFD_TIMERS_H
