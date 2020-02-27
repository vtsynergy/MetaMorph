/*
 * Exclusively for the core mechanics of timer implementations
 * None of this file should be active at all if WITH_TIMERS is not defined
 * i.e. it should only be #included in a conditional-compile block
 */

/** The top-level user APIs **/
#ifndef METAMORPH_TIMERS_H
#define METAMORPH_TIMERS_H

//Include metamorph.h to grab the properly-defined enum for modes
#ifndef METAMORPH_H
#include "metamorph.h"
#endif
#include <stdlib.h>

//Any special concerns for 
#ifdef WITH_CUDA
#endif


#ifdef WITH_OPENMP
#endif

//CUDA needs 2 events to use cudaEventElapsedTime
//OpenCL only needs one, if using the event returned from an API call
//OpenMP needs 2 to keep start/end time
/*
typedef union metaTimerEvent {
#ifdef WITH_CUDA
	cudaEvent_t cuda[2];
#endif
#ifdef WITH_OPENMP
	double openmp[2];
#endif
} metaTimerEvent;
*/
typedef meta_event metaTimerEvent;

typedef struct metaTimerQueueFrame {
	char const * name;
	metaTimerEvent event;
//Hijack meta_preferred_mode enum to advise the user of the frame/node how to interpret event
	meta_preferred_mode mode;
	size_t size;
//TODO add level 3 items
} metaTimerQueueFrame;

//TODO refactor code to use a frame internally, so that the frame can be changed
// without having to modify the QueueNode to match
typedef struct metaTimerQueueNode {
	char const * name;
	metaTimerEvent event;
//Hijack meta_preferred_mode enum to advise the user of the frame/node how to interpret event
	meta_preferred_mode mode;
	size_t size;
	struct metaTimerQueueNode * next;
} metaTimerQueueNode;

typedef struct metaTimerQueue {
	const char * name;
	metaTimerQueueNode * head, *tail;
} metaTimerQueue;

//A convenience structure to tie logically-named queues to possibly-changing
// indices in the Queue array. This way we can add a new type of timer
// (for a kernel or copy or whatever) and potentially change indices without
// needing to change a bunch of hardcoded constants.
enum metaTimerQueueEnum {
	//Host to Device copies into __constant memory (cudaMemcpyToSymbol and CL_MEM_READ_ONLY)
	c_D2H, c_H2D, c_H2H, c_D2D, c_H2Dc,
	k_reduce,
	k_dotProd,
	k_transpose_2d_face,
	k_pack_2d_face,
	k_unpack_2d_face,
	k_stencil_3d7p,
	k_csr,
	k_crc,
	//Special value used just to determine the size of the enum automagically
	// only works if we don't set explicit values for anything, and let it start from 0
	queue_count
};

a_err cl_get_event_node(metaTimerQueue * queue, char * ename,  metaTimerQueueFrame ** frame);
a_err metaTimerEnqueue(metaTimerQueueFrame * frame, metaTimerQueue * queue);
a_err metaTimerDequeue(metaTimerQueueFrame ** frame, metaTimerQueue * queue);
__attribute__((constructor(104))) a_err metaTimersInit();
a_err metaTimersFlush();
__attribute__((destructor(104))) a_err metaTimersFinish();

extern metaTimerQueue metaBuiltinQueues[];
#endif //METAMORPH_TIMERS_H
