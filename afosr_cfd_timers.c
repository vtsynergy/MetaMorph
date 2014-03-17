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

//Prepare the environment for timing, should be called by the first AFOSR Runtime
// Library Call, and should never be called again. If called a second time before
// accelTimersFinish is called, will be a silent NOOP.
a_err accelTimersInit() {

}

//Flush out properly-formatted timers/bandwidths/diagnostics
// Clears the timer queues entirely, and performs some (potentially intensive)
// aggregation of statistics, so should be used sparingly. If the application
// consists of large-scale sequential phases it might be appropriate to flush
// between phases, or if a sufficient number of library calls are performed to
// cause significant memory overhead (unlikely except in RAM-starved systems).
// Otherwise the accelTimersFlush() call inherent in accelTimersFinish should
// be sufficient.
a_err accelTimersFlush() {

}

//Safely flush timer stats to the output stream
a_err accelTimersFinish() {

	//first, make sure everything is flushed.
	accelTimersFlush();
	
	//then remove all reference points to these timers
	// (so that another thread can potentially spin up a separate new set..)
	//TODO timer cleanup
}

