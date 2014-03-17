#include "afosr_cfd_timers.h"

accelTimerQueue queue[queue_count];

//Take a copy of the frame and shove it on to the selected queue
// Do nothing else with the frame, the caller should handle releasing it
a_err accelTimerEnqueue(accelTimerQueueFrame * frame, accelTimerQueue * queue) {
	
}

//Take the copy of the front frame on the queue, and then remove it from the queue
// Do nothing else with the frame's copy, the caller should allocate and free it.
a_err accelTimerDequeue(accelTimerQueueFrame ** frame, accelTimerQueue * queue) {

}
