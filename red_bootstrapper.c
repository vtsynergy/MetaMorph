/* This is just the host side code required to invoke the reduction
 * It should not EVER need to know about CUDA/OpenCL/OpenMP
 */

#include <stdio.h>
#include <stdlib.h>
#include "afosr_cfd.h"


a_double * dev_data3;
a_double * dev_data3_2;
a_double * dev_data4;
a_double * reduction;
double * data3;
double * data4;
int ni, nj, nk, nm;

//      !This does the host and device data allocations.
	void data_allocate(int i, int j, int k, int m) {
            a_err istat = 0; 
            printf("ni:\t%d\n", ni); //print *,"ni:",ni
            printf("nj:\t%d\n", nj); //print *,"nj:",nj
            printf("nk:\t%d\n", nk); //print *,"nk:",nk
            data3 = (double *) malloc(sizeof(double)*ni*nj*nk);
	    data4 = (double *) malloc (sizeof(double)*ni*nj*nk*nm);
            printf("Status:\t%d\n", istat); //NOOP for output compatibility //print *,"Status:",istat
            istat = accel_alloc((void **) &dev_data3, sizeof(double)*ni*nj*nk);
            printf("Status:\t%d\n", istat);
	    istat = accel_alloc((void **) &dev_data3_2, sizeof(double)*ni*nj*nk);
            printf("Status:\t%d\n", istat);
	    istat = accel_alloc((void **) &dev_data4, sizeof(double)*ni*nj*nk*nm); 
            printf("Status:\t%d\n", istat);
	    istat = accel_alloc((void **) &reduction, sizeof(double));
            printf("Status:\t%d\n", istat);
            printf("Data Allocated\n"); 
      } 

//      !initilialize the host side data that has to be reduced here.
//      !For now I initialized it to 1.0
      void data_initialize() { 
	int i, j, k;
	for (i = ni*nj*nk*nm-1; i >= ni*nj*nk; i--) {
            data4[i] = 1.0;
	}
	for (; i >= 0; i--) {
	    data4[i] = 1.0;
            data3[i] = 1.0;
	}
	k = 0;
	for (; k < nk; k++) {
		j = 0;
		for (; j < nj; j++) {
			i = 0;
			for (; i < ni; i++) {
				if (i == 0 || j == 0 || k == 0) data3[i+j*ni+k*ni*nj] = 0.0f;
				if (i == ni-1 || j == nj-1 || k == nk-1) data3[i+j*ni+k*ni*nj] = 0.0f;
			}
		}
	}
      }

//      !Transfers data from host to device
      void data_transfer_h2d() {
	a_err ret= CL_SUCCESS; 
            ret |= accel_copy_h2d((void *) dev_data3, (void *) data3, sizeof(double)*ni*nj*nk, true);
            ret |= accel_copy_h2d((void *) dev_data4, (void *) data4, sizeof(double)*ni*nj*nk*nm, true);
            ret |= accel_copy_d2d((void *) dev_data3_2, (void *) dev_data3, sizeof(double)*ni*nj*nk, true);
} 

      void deallocate_() {
            accel_free(dev_data3); 
            free(data3); 
            accel_free(dev_data3_2); 
            accel_free(dev_data4);
            free(data4); 
	    accel_free(reduction); 
      } 
      void gpu_initialize() {

		//-1 is only supported with accelModePreferOpenCL
		// as a trigger to list all devices and select one
		//for CUDA use idevice = 0
            int istat, deviceused, idevice = -1; //integer::istat, deviceused, idevice

//            ! Initialize GPU
            istat = choose_accel(idevice, accelModePreferGeneric); //TODO make "choose_accel"

//            ! cudaChooseDevice
//            ! Tell me which GPU I use
		accel_preferred_mode mode;
            istat = get_accel(&deviceused, &mode); //TODO make "get_accel"
            printf("Device used\t%d\n", deviceused); //print *, 'Device used', deviceused
		

      } //end subroutine gpu_initialize

 int main(int argc, char **argv) { 
            int tx, ty, tz, gx, gy, gz, istat, i;
            a_dim3 dimgrid, dimblock, dimarray, arr_start, arr_end; //TODO move into CUDA backend, replace with generic struct
            char args[32];
            double sum_dot_gpu;
            double *dev_, *dev2; 
            i = argc; 
            if (i < 8) { 
                  printf("<ni><nj><nk><nm><tblockx><tblocky><tblockz>"); 
                  return(1); //stop
            } 
            ni = atoi(argv[1]);
            nj = atoi(argv[2]);
            nk = atoi(argv[3]);

            nm = atoi(argv[4]);

            tx = atoi(argv[5]);
            ty = atoi(argv[6]);
            tz = atoi(argv[7]);
		//TODO make timer initialization automatic
            #ifdef WITH_TIMERS
	    accelTimersInit();
	    #endif

            gpu_initialize(); //call gpu_initialize 
            data_allocate(ni,nj,nk,nm); //call data_allocate(ni,nj,nk,nm) 
            data_initialize(); //call data_initialize 
            data_transfer_h2d(); //call data_transfer_h2d 
            printf("Performing reduction\n"); //print *,'Performing reduction'
            //dev_data3 = 1.0
//!            tx = 8
//!            ty = 8
//!            tz = 2
//!            gx = 0 
//!            gy = 0
//!            gz = 0
            dimblock[0] = tx, dimblock[1] = ty, dimblock[2] = tz; // dimblock = {tx,ty,tz}; //TODO move into CUDA backend, replace with generic struct
            printf("ni:\t%d\n", ni); //print *,"ni:",ni
            printf("nj:\t%d\n", nj); //print *,"nj:",nj
            printf("nk:\t%d\n", nk); //print *,"nk:",nk
            printf("gyr:\t%d\n", (nj-2)%ty); //print *,"gyr:",modulo(ni-2,ty)

            printf("gxr:\t%d\n", (ni-2)%tx); //print *,"gxr:",modulo(ni-2,tx)
            printf("gzr:\t%d\n", (nk-2)%tz); //print *,"gzr:",modulo(nk-2,tz)
            if ((nj-2)%ty != 0)  //if(modulo(ni-2,ty).ne.0)then
                  gy = (nj-2)/ty +1;
            else
                  gy = (nj-2)/ty;
            //end if
            if ((ni-2)%tx != 0) //if(modulo(nj-2,tx).ne.0)then
                  gx = (ni-2)/tx +1;
            else
                  gx = (ni-2)/tx;
            //end if
            if ((nk-2)%tz != 0) //if(modulo(nk-2,tz).ne.0)then
                  gz = (nk-2)/tz +1;
            else
                  gz = (nk-2)/tz;
            //end if
	    //CUDA doesn't support dimgrid[2] != 1, but we use this to pass the number of slab iterations the kernel needs to run internally
            dimgrid[0] = gx, dimgrid[1] = gy, dimgrid[2] = gz; //dimgrid = {gx,gy,1}; // TODO move into CUDA backend, replace with generic struct
            printf("gx:\t%d\n", gx); //print *,"gx:",gx
            printf("gy:\t%d\n", gy); //print *,"gy:",gy
            printf("gz:\t%d\n", gz); //print *,"gz:",gz
		double zero = 0.0;
	    dimarray[0] = ni, dimarray[1] = nj, dimarray[2] = nk;
	    arr_start[0] = arr_start[1] = arr_start[2] = 1;
	    arr_end[0] = ni-2, arr_end[1] = nj-2, arr_end[2] = nk-2;
for (i = 0; i < 10; i++) { //do i=1,10
	istat =	accel_copy_h2d((void *) reduction, (void *) &zero, sizeof(double), true);
		//Validate grid and block sizes (if too big, shrink the z-dim and add iterations)
		for(;accel_validate_worksize(&dimgrid, &dimblock) != 0 && dimblock[2] > 1; dimgrid[2] <<=1, dimblock[2] >>=1);
		// Z-scaling won't be enough, abort
		//TODO: Implement a way to do Y- and X-scaling
		if (accel_validate_worksize(&dimgrid, &dimblock)) {

		}
		

		//Call the entire reduction
		a_err ret = accel_dotProd(&dimgrid, &dimblock, dev_data3, dev_data3_2, &dimarray, &arr_start, &arr_end, reduction, true);
		fprintf(stderr, "Kernel Status: %d\n", ret);

//           kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*sizeof(double)>>>(dev_data3, //call kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*8>>>(dev_data3 & //TODO move into CUDA backend, make "accel_reduce"
//           dev_data3_2, ni, nj, nk, 2, 2, 2, nj-1, ni-1, nk-1, gz, reduction, tx*ty*tz); //& ,dev_data3_2,ni,nj,nk,2,2,2,nj-1,ni-1,nk-1,gz,reduction,tx*ty*tz) //TODO - see previous
//            istat = cudaThreadSynchronize(); //cudathreadsynchronize()// TODO move into CUDA backend
	//	printf("cudaThreadSynchronize error code:%d\n", istat);            
		istat = accel_copy_d2h((void *) &sum_dot_gpu, (void *) reduction, sizeof(double), false);
            printf("Test Reduction:\t%f\n", sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
            //printf("Test Reduction:\t%d\n", sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	    //accelTimersFlush();
            } //end do
            deallocate_(); //call deallocate_i
	    #ifdef WITH_TIMERS
	    accelTimersFinish();
	    #endif
      } //end program main
