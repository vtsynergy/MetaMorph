/* This is just the host side code required to invoke the reduction
 * It should not EVER need to know about CUDA/OpenCL/OpenMP
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "metamorph.h"

//#define DATA4

//global for the current type
meta_type_id g_type;
size_t g_typesize;

//Sets the benchmark's global configuration for one of the supported data types
void set_type(meta_type_id type) {
	switch (type) {
		case a_db:
		g_typesize=sizeof(double);
		break;

		case a_fl:
		g_typesize=sizeof(float);
		break;

		case a_ul:
		g_typesize=sizeof(unsigned long);
		break;

		case a_in:
		g_typesize=sizeof(int);
		break;

		case a_ui:
		g_typesize=sizeof(unsigned int);
		break;

		default:
		fprintf(stderr, "Unsupported type: [%d] provided to set_type!\n", type);
		exit(-1);
		break;
	}
	g_type = type;
}

//device buffers
void * dev_data3, *dev_data3_2, *reduction;

#ifdef DATA4
void *dev_data4;
#endif

// host buffers
void *data3;

#ifdef DATA4
void *data4;
#endif

int ni, nj, nk, nm, iters;

//      !This does the host and device data allocations.
	void data_allocate(int i, int j, int k, int m) {

        a_err istat = 0;
        //printf("ni:\t%d\n", ni); //print *,"ni:",ni
        //printf("nj:\t%d\n", nj); //print *,"nj:",nj
        //printf("nk:\t%d\n", nk); //print *,"nk:",nk
        data3 = malloc(g_typesize*ni*nj*nk);
#ifdef DATA4
	    data4 = malloc(g_typesize*ni*nj*nk*nm);
#endif
        //printf("Status:\t%d\n", istat); //NOOP for output compatibility //print *,"Status:",istat
#ifndef UNIFIED_MEMORY
	    istat = meta_alloc( &dev_data3, g_typesize*ni*nj*nk);
#endif
        //printf("Status:\t%d\n", istat);
	    istat = meta_alloc( &dev_data3_2, g_typesize*ni*nj*nk);
	    //istat = meta_alloc( &dev_data3cp, g_typesize*ni*nj*nk);
        //printf("Status:\t%d\n", istat);
#ifdef DATA4
	    istat = meta_alloc( &dev_data4, g_typesize*ni*nj*nk*nm); 
#endif
        //printf("Status:\t%d\n", istat);
	    istat = meta_alloc( &reduction, g_typesize);
        //printf("Status:\t%d\n", istat);
        //printf("Data Allocated\n");
      } 

//      !initilialize the host side data that has to be reduced here.
//      !For now I initialized it to 1.0
      void data_initialize() {
	int i, j, k;
	switch(g_type) {
		default:
		case a_db:
			{
				double * l_data3 = (double *) data3;
#ifdef DATA4
				double * l_data4 = (double *) data4;
#endif
				for (i = ni*nj*nk*nm-1; i >= ni*nj*nk; i--) {
#ifdef DATA4
			            l_data4[i] = 1;
#endif
				}
				for (; i >= 0; i--) {
#ifdef DATA4
				    l_data4[i] = 1;
#endif
				    l_data3[i] = 1;
				}
				k = 0;
				for (; k < nk; k++) {
					j = 0;
					for (; j < nj; j++) {
						i = 0;
						for (; i < ni; i++) {
							if (i == 0 || j == 0 || k == 0) l_data3[i+j*ni+k*ni*nj] = 0;
							if (i == ni-1 || j == nj-1 || k == nk-1) l_data3[i+j*ni+k*ni*nj] = 0;
						}
					}
				}
			}
		break;
		
		case a_fl:
			{
				float * l_data3 = (float *) data3;
#ifdef DATA4
				float * l_data4 = (float *) data4;
#endif
				for (i = ni*nj*nk*nm-1; i >= ni*nj*nk; i--) {
#ifdef DATA4
			            l_data4[i] = 1;
#endif
				}
				for (; i >= 0; i--) {
#ifdef DATA4
				    l_data4[i] = 1;
#endif
			        l_data3[i] = 1;
				}
				k = 0;
				for (; k < nk; k++) {
					j = 0;
					for (; j < nj; j++) {
						i = 0;
						for (; i < ni; i++) {
							if (i == 0 || j == 0 || k == 0) l_data3[i+j*ni+k*ni*nj] = 0;
							if (i == ni-1 || j == nj-1 || k == nk-1) l_data3[i+j*ni+k*ni*nj] = 0;
						}
					}
				}
			}
		break;
		
		case a_ul: 
			{
				unsigned long * l_data3 = (unsigned long *) data3;
#ifdef DATA4
				unsigned long * l_data4 = (unsigned long *) data4;
#endif
				for (i = ni*nj*nk*nm-1; i >= ni*nj*nk; i--) {
#ifdef DATA4
			         l_data4[i] = 1;
#endif
				}
				for (; i >= 0; i--) {
#ifdef DATA4
				    l_data4[i] = 1;
#endif
				    l_data3[i] = 1;
				}
				k = 0;
				for (; k < nk; k++) {
					j = 0;
					for (; j < nj; j++) {
						i = 0;
						for (; i < ni; i++) {
							if (i == 0 || j == 0 || k == 0) l_data3[i+j*ni+k*ni*nj] = 0;
							if (i == ni-1 || j == nj-1 || k == nk-1) l_data3[i+j*ni+k*ni*nj] = 0;
						}
					}
				}
			}
		break;
		
		case a_in:
			{
				int * l_data3 = (int *) data3;
#ifdef DATA4
				int * l_data4 = (int *) data4;
#endif
				for (i = ni*nj*nk*nm-1; i >= ni*nj*nk; i--) {
#ifdef DATA4
			            l_data4[i] = 1;
#endif
				}
				for (; i >= 0; i--) {
#ifdef DATA4
				    l_data4[i] = 1;
#endif
			        l_data3[i] = 1;
				}
				k = 0;
				for (; k < nk; k++) {
					j = 0;
					for (; j < nj; j++) {
						i = 0;
						for (; i < ni; i++) {
							if (i == 0 || j == 0 || k == 0) l_data3[i+j*ni+k*ni*nj] = 0;
							if (i == ni-1 || j == nj-1 || k == nk-1) l_data3[i+j*ni+k*ni*nj] = 0;
						}
					}
				}
			}
		break;
		
		case a_ui:
			{
				unsigned int * l_data3 = (unsigned int *) data3;
#ifdef DATA4
				unsigned int * l_data4 = (unsigned int *) data4;
#endif
				for (i = ni*nj*nk*nm-1; i >= ni*nj*nk; i--) {
#ifdef DATA4
			            l_data4[i] = 1;
#endif
				}
				for (; i >= 0; i--) {
#ifdef DATA4
				    l_data4[i] = 1;
#endif
			        l_data3[i] = 1;
				}
				k = 0;
				for (; k < nk; k++) {
					j = 0;
					for (; j < nj; j++) {
						i = 0;
						for (; i < ni; i++) {
							if (i == 0 || j == 0 || k == 0) l_data3[i+j*ni+k*ni*nj] = 0;
							if (i == ni-1 || j == nj-1 || k == nk-1) l_data3[i+j*ni+k*ni*nj] = 0;
						}
					}
				}
			}
		break;
		
	}
      }


//      !Transfers data from host to device
      void data_transfer_h2d() {
	a_err ret;
#ifdef WITH_CUDA
	ret= cudaSuccess;
#endif

#ifdef WITH_OPENCL
	ret= CL_SUCCESS; 
#endif

#ifdef WITH_OPENMP
	ret= 0;
#endif
	//TODO add timing loops
	int iter;
	struct timeval start, end;

#ifndef UNIFIED_MEMORY
	gettimeofday(&start, NULL);
	for (iter = 0; iter < iters; iter++)
            ret |= meta_copy_h2d( dev_data3, data3, g_typesize*ni*nj*nk, false);
	gettimeofday(&end, NULL);
	printf("D2H time: %f\n", ((end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec))/(iters));
#else
	dev_data3 = data3;
#endif

#ifdef DATA4
            ret |= meta_copy_h2d( dev_data4, data4, g_typesize*ni*nj*nk*nm, false);
#endif

	gettimeofday(&start, NULL);
	for (iter = 0; iter < iters; iter++)
            ret |= meta_copy_d2d( dev_data3_2, dev_data3, g_typesize*ni*nj*nk, false);
	gettimeofday(&end, NULL);
	printf("D2D time: %f\n", ((end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec))/(iters));
	
} 

      void deallocate_() {
#ifndef UNIFIED_MEMORY
            meta_free(dev_data3); 
#endif
            free(data3); 
            meta_free(dev_data3_2); 
#ifdef DATA4
            meta_free(dev_data4);
            free(data4); 
#endif
	    meta_free(reduction); 
      } 


      void gpu_initialize() {

		//-1 is only supported with metaModePreferOpenCL
		// as a trigger to list all devices and select one
		//for CUDA use idevice = 0
            int istat, deviceused; //integer::istat, deviceused, idevice

		int idevice;
//            ! Initialize GPU
#ifdef WITH_CUDA
            idevice = 0;
            istat = choose_accel(idevice, metaModePreferCUDA); //TODO make "choose_accel"
#endif

#ifdef WITH_OPENCL
            idevice = -1;
            istat = choose_accel(idevice, metaModePreferGeneric); //TODO make "choose_accel"
#endif

#ifdef WITH_OPENMP
            idevice = 0;
            istat = choose_accel(idevice, metaModePreferOpenMP); //TODO make "choose_accel"
#endif

//            ! cudaChooseDevice
//            ! Tell me which GPU I use
		meta_preferred_mode mode;
            istat = get_accel(&deviceused, &mode); //TODO make "get_accel"
 //           printf("Device used\t%d\n", deviceused); //print *, 'Device used', deviceused
		

      } //end subroutine gpu_initialize

      void print_grid(double * grid) {
      	int i,j,k;
      	for (k = 0; k < nk; k++) {
      		for (j =0; j < nj; j++) {
      			for (i=0; i < ni; i++) {
      				printf("[%f] ", grid[i+ j*(ni) + k*nj*(ni)]);
      			}
      			printf("\n");
      		}
      		printf("\n");
      	}
      }

 int main(int argc, char **argv) { 
            int tx, ty, tz, gx, gy, gz, istat, i, l_type;
            a_dim3 dimgrid, dimblock, dimarray, arr_start, arr_end; //TODO move into CUDA backend, replace with generic struct
            a_dim3 trans_2d;
            char args[32];

            i = argc; 
            if (i < 10) { 
                  printf("<ni><nj><nk><nm><tblockx><tblocky><tblockz><type><iters>"); 
                  return(1); //stop
            } 
            ni = atoi(argv[1]);
            nj = atoi(argv[2]);
            nk = atoi(argv[3]);

            nm = atoi(argv[4]);

            tx = atoi(argv[5]);
            ty = atoi(argv[6]);
            tz = atoi(argv[7]);

	    l_type = atoi(argv[8]);
	    set_type((meta_type_id)l_type);
	    iters = atoi(argv[9]);
//For simplicity when testing across all types, these are kept as void
// and explicitly cast for the few calls they are necessary, based on g_type
            void * sum_dot_gpu, * zero;
		sum_dot_gpu = malloc(g_typesize);
		zero = malloc(g_typesize);
		//TODO make timer initialization automatic
            #ifdef WITH_TIMERS
	    metaTimersInit();
	    #endif

            gpu_initialize(); //call gpu_initialize 
            data_allocate(ni,nj,nk,nm); //call data_allocate(ni,nj,nk,nm) 
            data_initialize(); //call data_initialize 

            data_transfer_h2d(); //call data_transfer_h2d 
            printf("Performing dot-product, type %d\n", l_type); //print *,'Performing reduction'
#ifdef WITH_OPENMP
			#pragma omp parallel
			{
				#pragma omp master
				printf("num threads %d\n", omp_get_num_threads());
			}
#endif
            //dev_data3 = 1.0
//!            tx = 8
//!            ty = 8
//!            tz = 2
//!            gx = 0 
//!            gy = 0
//!            gz = 0
            dimblock[0] = tx, dimblock[1] = ty, dimblock[2] = tz; // dimblock = {tx,ty,tz}; //TODO move into CUDA backend, replace with generic struct
            //printf("ni:\t%d\n", ni); //print *,"ni:",ni
            //printf("nj:\t%d\n", nj); //print *,"nj:",nj
            //printf("nk:\t%d\n", nk); //print *,"nk:",nk
            //printf("gyr:\t%d\n", (nj-2)%ty); //print *,"gyr:",modulo(ni-2,ty)

            //printf("gxr:\t%d\n", (ni-2)%tx); //print *,"gxr:",modulo(ni-2,tx)
            //printf("gzr:\t%d\n", (nk-2)%tz); //print *,"gzr:",modulo(nk-2,tz)
            if ((nj)%ty != 0)  //if(modulo(ni-2,ty).ne.0)then
                  gy = (nj)/ty +1;
            else
                  gy = (nj)/ty;
            //end if
            if ((ni)%tx != 0) //if(modulo(nj-2,tx).ne.0)then
                  gx = (ni)/tx +1;
            else
                  gx = (ni)/tx;
            //end if
            if ((nk)%tz != 0) //if(modulo(nk-2,tz).ne.0)then
                  gz = (nk)/tz +1;
            else
                  gz = (nk)/tz;
            //end if
	    //CUDA doesn't support dimgrid[2] != 1, but we use this to pass the number of slab iterations the kernel needs to run internally
            dimgrid[0] = gx, dimgrid[1] = gy, dimgrid[2] = gz; //dimgrid = {gx,gy,1}; // TODO move into CUDA backend, replace with generic struct
            //printf("gx:\t%d\n", gx); //print *,"gx:",gx
            //printf("gy:\t%d\n", gy); //print *,"gy:",gy
            //printf("gz:\t%d\n", gz); //print *,"gz:",gz
switch(g_type) {
	case a_db:
		*(double*)zero = 0;
	break;

	case a_fl:
		*(float*)zero = 0;
	break;

	case a_ul:
		*(unsigned long*)zero = 0;
	break;

	case a_in:
		*(int *)zero = 0;
	break;

	case a_ui:
		*(unsigned int *)zero = 0;
	break;
}
	    dimarray[0] = ni, dimarray[1] = nj, dimarray[2] = nk;
	    arr_start[0] = arr_start[1] = arr_start[2] = 0;
	    arr_end[0] = ni-1, arr_end[1] = nj-1, arr_end[2] = nk-1;
	    trans_2d[0] = ni;
	    trans_2d[1] = nj * nk;
	    trans_2d[2] = 1;
//for (i = 0; i < 1; i++) { //do i=1,10
	istat =	meta_copy_h2d( reduction, zero, g_typesize, true);
		//Validate grid and block sizes (if too big, shrink the z-dim and add iterations)
		for(;meta_validate_worksize(&dimgrid, &dimblock) != 0 && dimblock[2] > 1; dimgrid[2] <<=1, dimblock[2] >>=1);
		// Z-scaling won't be enough, abort
		//TODO: Implement a way to do Y- and X-scaling
		if (meta_validate_worksize(&dimgrid, &dimblock)) {

		}

		//Call the entire reduction
		//TODO add timer loop
		int iter;
		struct timeval start, end;
		a_err ret;

		gettimeofday(&start, NULL);
		for (iter = 0; iter < iters; iter++)
			ret = meta_dotProd(&dimgrid, &dimblock, dev_data3, dev_data3_2, &dimarray, &arr_start, &arr_end, reduction, g_type, false);
		gettimeofday(&end, NULL);
		fprintf(stderr, "Kernel Status: %d\n", ret);
		printf("Kern time: %f\n", ((end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec))/(iters));

//           kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*sizeof(double)>>>(dev_data3, //call kernel_reduction3<<<dimgrid,dimblock,tx*ty*tz*8>>>(dev_data3 & //TODO move into CUDA backend, make "meta_reduce"
//           dev_data3_2, ni, nj, nk, 2, 2, 2, nj-1, ni-1, nk-1, gz, reduction, tx*ty*tz); //& ,dev_data3_2,ni,nj,nk,2,2,2,nj-1,ni-1,nk-1,gz,reduction,tx*ty*tz) //TODO - see previous
//            istat = cudaThreadSynchronize(); //cudathreadsynchronize()// TODO move into CUDA backend
	//	printf("cudaThreadSynchronize error code:%d\n", istat);            
		istat = meta_copy_d2h(sum_dot_gpu, reduction, g_typesize, false);

switch(g_type) {
	case a_db:
		printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%f]\n", (*(double*)sum_dot_gpu == (double)((ni-2)*(nj-2)*(nk-2)*iters) ? "PASSED" : "FAILED"), (ni-2)*(nj-2)*(nk-2)*iters, (*(double*)sum_dot_gpu)); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_fl:
		printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%f]\n", (*(float*)sum_dot_gpu == (float)((ni-2)*(nj-2)*(nk-2)*iters) ? "PASSED" : "FAILED"), (ni-2)*(nj-2)*(nk-2)*iters, (*(float*)sum_dot_gpu)); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_ul:
		printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%ld]\n", (*(unsigned long*)sum_dot_gpu == (unsigned long)((ni-2)*(nj-2)*(nk-2)*iters) ? "PASSED" : "FAILED"), (ni-2)*(nj-2)*(nk-2)*iters, (*(unsigned long*)sum_dot_gpu)); //print *, "Test Reduction:",sum_dot_gpu
		printf("Test Dot-Product:\t%lu\n", *(unsigned long*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_in:
		printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%d]\n", (*(int*)sum_dot_gpu == (int)((ni-2)*(nj-2)*(nk-2)*iters) ? "PASSED" : "FAILED"), (ni-2)*(nj-2)*(nk-2)*iters, (*(int*)sum_dot_gpu)); //print *, "Test Reduction:",sum_dot_gpu
		printf("Test Dot-Product:\t%d\n", *(int*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_ui:
		printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%d]\n", (*(unsigned int*)sum_dot_gpu == (unsigned int)((ni-2)*(nj-2)*(nk-2)*iters) ? "PASSED" : "FAILED"), (ni-2)*(nj-2)*(nk-2)*iters, (*(unsigned int*)sum_dot_gpu)); //print *, "Test Reduction:",sum_dot_gpu
		printf("Test Dot-Product:\t%d\n", *(unsigned int*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;
}

	gettimeofday(&start, NULL);
	for (iter = 0; iter < iters; iter++)
		ret = meta_stencil_3d7p(&dimgrid, &dimblock, dev_data3, dev_data3_2, &dimarray, &arr_start, &arr_end, g_type, false);
	gettimeofday(&end, NULL);
	fprintf(stderr, "Kernel Status: %d\n", ret);
	printf("stencil_3d7p Kern time: %f\n", ((end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec))/(iters));
	//print_grid(dev_data3_2);
	istat =	meta_copy_h2d( reduction, zero, g_typesize, true);
	ret = meta_reduce(&dimgrid, &dimblock, dev_data3_2, &dimarray, &arr_start, &arr_end, reduction, g_type, false);
	istat = meta_copy_d2h(sum_dot_gpu, reduction, g_typesize, false);

	switch(g_type) {
	case a_db:
		printf("Test stencil_3d7p:\t%f\n", *(double*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_fl:
		printf("Test stencil_3d7p:\t%f\n", *(float*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_ul:
		printf("Test stencil_3d7p:\t%lu\n", *(unsigned long*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_in:
		printf("Test stencil_3d7p:\t%d\n", *(int*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;

	case a_ui:
		printf("Test Dot-Product:\t%d\n", *(unsigned int*)sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	break;
	}

	gettimeofday(&start, NULL);
	for (iter = 0; iter < iters; iter++)
		//ret = meta_transpose_2d_face(&dimgrid, &dimblock, dev_data3cp, dev_data3, &trans_2d, &trans_2d,  g_type, false);
		ret = meta_transpose_2d_face(&dimgrid, &dimblock, dev_data3_2, dev_data3, &trans_2d, &trans_2d,  g_type, false);
	gettimeofday(&end, NULL);
	fprintf(stderr, "transpose Kernel Status: %d\n", ret);
	printf("transpose Kern time: %f\n", ((end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec))/(iters));
    //ret = meta_copy_d2d( dev_data3_2, dev_data3, g_typesize*ni*nj*nk, false);

	//TODO add a copy-back timer loop
	gettimeofday(&start, NULL);
	for (iter = 0; iter < iters; iter++)
		meta_copy_d2h(data3, dev_data3, g_typesize*ni*nj*nk, false);
	gettimeofday(&end, NULL);
	printf("D2H time: %f\n", ((end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec))/(iters));

	    //printf("Test Reduction:\t%d\n", sum_dot_gpu); //print *, "Test Reduction:",sum_dot_gpu
	    //metaTimersFlush();
//            } //end do
            deallocate_(); //call deallocate_i
	    #ifdef WITH_TIMERS
	    metaTimersFinish();
	    #endif
	    return 0;
      } //end program main
