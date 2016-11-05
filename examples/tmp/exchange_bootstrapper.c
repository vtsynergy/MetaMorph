/*
 * Simple MPI-enabled bootstrapper designed to test transpose, pack/unpack
 and send/recv functionality.

 Allocates a 3D grid, fills various faces, and tests operations on them
 by taking sub-sums which should match easy-to-compute values.

 Transpose is tested by filling all rows of a face with their row indices,
 transposing the face, summing all elements on a transposed row, and
 comparing the value to the directly computed sum of all row indices.
 i.e. (n^2 + n)/2

 Pack/unpack/exchange are tested by taking the same (untransposed) face and
 computing 4 sums: (1) All elements on the face, (2) packing the face to a
 contiguous buffer and summing all vector elements, (3) receiving the packed
 face and summing all vector elements, and (4) unpacking the face and summing
 all elements of the face. All four sums should match each other, and the
 directly computed sum given by: (xdim)*((ydim^2 + ydim)/2).

 Both transpose and pack/unpack/exchange tests should be runnable on any face
 of the 3D space. In this two-process test, connectivity is toroidial in all
 directions. i.e. Proc1 is the north, south, east, west, top, and bottom
 neighbor of Proc0 and vice versa.
 */

#include <stdio.h>
#include <stdlib.h>
#include "metamorph.h"

//Checks the sum of a face based on the formula:
// xdim*ydim * (zoffset + (xdim+ydim)/2 - 1)
// where a = xdim, b = ydim, and c = zoffset
#define SUM_FACE(a,b,c) ((((a)*(b))*((c)+(((a)+(b))/2.0)-1)))

//global for the current type
meta_type_id g_type;
size_t g_typesize;

//Sets the benchmark's global configuration for one of the supported data types
void set_type(meta_type_id type) {
	switch (type) {
	case a_db:
		g_typesize = sizeof(double);
		break;

	case a_fl:
		g_typesize = sizeof(float);
		break;

	case a_ul:
		g_typesize = sizeof(unsigned long);
		break;

	case a_in:
		g_typesize = sizeof(int);
		break;

	case a_ui:
		g_typesize = sizeof(unsigned int);
		break;

	default:
		fprintf(stderr, "Unsupported type: [%d] provided to set_type!\n", type);
		exit(-1);
		break;
	}
	g_type = type;
}

void *data3, *face[6];
void *reduction, *dev_data3, *dev_face[6];

void data_allocate(int i, int j, int k) {
	a_err istat = 0;
	//Reduction sum
	istat = meta_alloc(&reduction, g_typesize);
	printf("Status (Sum alloc):\t%d\n", istat);
	//3D region
	data3 = malloc(g_typesize * i * j * k);
	istat = meta_alloc(&dev_data3, g_typesize * i * j * k);
	printf("Status (3D alloc):\t%d\n", istat);
	//North/south faces
	face[0] = malloc(g_typesize * i * j);
	face[1] = malloc(g_typesize * i * j);
	istat = meta_alloc(&dev_face[0], g_typesize * i * j);
	istat |= meta_alloc(&dev_face[1], g_typesize * i * j);
	printf("Status (N/S faces):\t%d\n", istat);
	//East/west faces
	face[2] = malloc(g_typesize * j * k);
	face[3] = malloc(g_typesize * j * k);
	istat = meta_alloc(&dev_face[2], g_typesize * j * k);
	istat |= meta_alloc(&dev_face[3], g_typesize * j * k);
	printf("Status (E/W faces):\t%d\n", istat);
	//Top/bottom faces
	face[4] = malloc(g_typesize * i * k);
	face[5] = malloc(g_typesize * i * k);
	istat = meta_alloc(&dev_face[4], g_typesize * i * k);
	istat |= meta_alloc(&dev_face[5], g_typesize * i * k);
	printf("Status (T/B faces):\t%d\n", istat);
	printf("Data Allocated\n");
}

void data_initialize(int ni, int nj, int nk) {
	int i, j, k;
	switch (g_type) {
	default:
	case a_db: {
		double *l_data3 = (double *) data3;
		for (i = ni - 1; i >= 0; i--) {
			for (j = nj - 1; j >= 0; j--) {
				for (k = nk - 1; k >= 0; k--) {
//								printf("HELP: %d - %d %d %d %d %d %d\n", i+j*ni+k*ni*nj, i, j, k, ni, nj, nk);
					if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
							|| k == nk - 1) {
						l_data3[i + j * ni + k * ni * nj] = i + j + k;
					} else {
						l_data3[i + j * ni + k * ni * nj] = 0.0f;
					}
				}
			}
		}
	}
		break;

	case a_fl: {
		float *l_data3 = (float *) data3;
		for (i = ni - 1; i >= 0; i--) {
			for (j = nj - 1; j >= 0; j--) {
				for (k = nk - 1; k >= 0; k--) {
					if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
							|| k == nk - 1) {
						l_data3[i + j * ni + k * ni * nj] = i + j + k;
					} else {
						l_data3[i + j * ni + k * ni * nj] = 0.0f;
					}
				}
			}
		}
	}
		break;

	case a_ul: {
		unsigned long *l_data3 = (unsigned long *) data3;
		for (i = ni - 1; i >= 0; i--) {
			for (j = nj - 1; j >= 0; j--) {
				for (k = nk - 1; k >= 0; k--) {
					if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
							|| k == nk - 1) {
						l_data3[i + j * ni + k * ni * nj] = i + j + k;
					} else {
						l_data3[i + j * ni + k * ni * nj] = 0;
					}
				}
			}
		}
	}
		break;

	case a_in: {
		int *l_data3 = (int *) data3;
		for (i = ni - 1; i >= 0; i--) {
			for (j = nj - 1; j >= 0; j--) {
				for (k = nk - 1; k >= 0; k--) {
					if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
							|| k == nk - 1) {
						l_data3[i + j * ni + k * ni * nj] = i + j + k;
					} else {
						l_data3[i + j * ni + k * ni * nj] = 0;
					}
				}
			}
		}
	}
		break;

	case a_ui: {
		unsigned int *l_data3 = (unsigned int *) data3;
		for (i = ni - 1; i >= 0; i--) {
			for (j = nj - 1; j >= 0; j--) {
				for (k = nk - 1; k >= 0; k--) {
					if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
							|| k == nk - 1) {
						l_data3[i + j * ni + k * ni * nj] = i + j + k;
					} else {
						l_data3[i + j * ni + k * ni * nj] = 0;
					}
				}
			}
		}
	}
		break;

	}
}

void gpu_initialize(int rank) {

	//-1 is only supported with metaModePreferOpenCL
	// as a trigger to list all devices and select one
	//for CUDA use idevice = 0
	int istat, deviceused, idevice = rank; //integer::istat, deviceused, idevice

//            ! Initialize GPU
	//istat = meta_set_acc(idevice, (rank & 1 ? metaModePreferCUDA : metaModePreferOpenCL)); //TODO make "meta_set_acc"
	//istat = meta_set_acc(idevice, (rank & 1 ? metaModePreferOpenCL : metaModePreferCUDA)); //TODO make "meta_set_acc"
	istat = meta_set_acc(idevice, metaModePreferCUDA); //TODO make "meta_set_acc"3
	//istat = meta_set_acc(idevice, metaModePreferOpenCL);
	//istat = meta_set_acc(idevice, metaModePreferGeneric); //TODO make "meta_set_acc"

//            ! cudaChooseDevice
//            ! Tell me which GPU I use
	meta_preferred_mode mode;
	istat = meta_get_acc(&deviceused, &mode); //TODO make "meta_get_acc"
	printf("Device used\t%d\n", deviceused); //print *, 'Device used', deviceused
	//Test	

} //end subroutine gpu_initialize
int check_face_sum(void * sum, int a, int b, int c) {
	printf("CHECK: %d %d %d\n", a, b, c);
	int ret = 0;
	switch (g_type) {
	case a_db:
		if (SUM_FACE(a,b,c) != *(double *) sum) {
			fprintf(stderr,
					"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%f]\n",
					(double) SUM_FACE(a, b, c), *(double *) sum);
			ret = -1;
		}
		break;
	case a_fl:
		if (SUM_FACE(a,b,c) != *(float *) sum) {
			fprintf(stderr,
					"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%f]\n",
					(float) SUM_FACE(a, b, c), *(float *) sum);
			ret = -1;
		}
		break;
	case a_ul:
		if (SUM_FACE(a,b,c) != (float) (*(unsigned long *) sum)) {
			fprintf(stderr,
					"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%lu]\n",
					SUM_FACE(a, b, c), *(unsigned long *) sum);
			ret = -1;
		}
		break;
	case a_in:
		if (SUM_FACE(a,b,c) != (float) (*(int *) sum)) {
			fprintf(stderr,
					"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%d]\n",
					SUM_FACE(a, b, c), *(int *) sum);
			ret = -1;
		}
		break;
	case a_ui:
		if (SUM_FACE(a,b,c) != (float) (*(unsigned int *) sum)) {
			fprintf(stderr,
					"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%d]\n",
					SUM_FACE(a, b, c), *(unsigned int *) sum);
			ret = -1;
		}
		break;
	default:
		//well if the type isn't one of these 5, it should fail
		ret = -1;
		break;

	}
	return ret;
}

//Workhorse for computing a 2.5D slab from 3D grid
meta_face * make_slab2d_from_3d(int face, int ni, int nj, int nk,
		int thickness) {
	meta_face * ret = (meta_face*) malloc(
			sizeof(meta_face));
	ret->count = 3;
	ret->size = (int*) malloc(sizeof(int) * 3);
	ret->stride = (int*) malloc(sizeof(int) * 3);
	//all even faces start at the origin, all others start at some offset
	// defined by the dimensions of the prism
	if (face & 1) {
		if (face == 1)
			ret->start = ni * nj * (nk - thickness);
		if (face == 3)
			ret->start = ni - thickness;
		if (face == 5)
			ret->start = ni * (nj - thickness);
	} else
		ret->start = 0;
	ret->size[0] = nk, ret->size[1] = nj, ret->size[2] = ni;
	if (face < 2)
		ret->size[0] = thickness;
	if (face > 3)
		ret->size[1] = thickness;
	if (face > 1 && face < 4)
		ret->size[2] = thickness;
	ret->stride[0] = ni * nj, ret->stride[1] = ni, ret->stride[2] = 1;
	printf(
			"Generated Face:\n\tcount: %d\n\tstart: %d\n\tsize: %d %d %d\n\tstride: %d %d %d\n",
			ret->count, ret->start, ret->size[0], ret->size[1], ret->size[2],
			ret->stride[0], ret->stride[1], ret->stride[2]);
	return ret;
}
//FIXME: Left for compatibility, remove once dependencies are resolved
meta_face * make_face(int face, int ni, int nj, int nk) {
	return make_slab2d_from_3d(face, ni, nj, nk, 1);
}

void check_dims(a_dim3 dim, a_dim3 s, a_dim3 e) {
	printf(
			"Integrity check dim(%ld, %ld, %ld) start(%ld, %ld, %ld) end(%ld, %ld, %ld)\n",
			dim[0], dim[1], dim[2], s[0], s[1], s[2], e[0], e[1], e[2]);

}

void check_buffer(void* h_buf, void * d_buf, int leng) {
	meta_copy_d2h(h_buf, d_buf, g_typesize * leng, 0);
	int i;
	double sum = 0.0;
	for (i = 0; i < leng; i++) {
		printf("%f\n", ((double*) h_buf)[i]);
		sum += ((double*) h_buf)[i];
	}
	printf("SUM: %f\n", sum);

}

//Returns 1 if the relative difference between expect and test is within tol
//Returns 0 otherwise
int check_fp(double expect, double test, double tol) {
	return (abs((expect - test) / expect) < tol);
}

int main(int argc, char **argv) {
	meta_mpi_init(&argc, &argv);

#ifdef __DEBUG__
	int breakMe = 0;
	while (breakMe);
#endif

	/*{
	 int i = 0;
	 char hostname[256];
	 gethostname(hostname, sizeof(hostname));
	 printf("PID %d on %s ready for attach\n", getpid(), hostname);
	 fflush(stdout);
	 while (0 == i)
	 sleep(5);
	 }*/
	int comm_sz;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("Hello from rank %d!\n", rank);
	int i = argc;
	int ni, nj, nk, tx, ty, tz, face_id, l_type;
	a_bool async, autoconfig;
	meta_face * face_spec;

	a_dim3 dimgrid_red, dimblock_red, dimgrid_tr_red, dimarray_3d, arr_start,
			arr_end, dim_array2d, start_2d, end_2d, trans_dim, rtrans_dim;
	if (i < 11) {
		printf(
				"<ni> <nj> <nk> <tblockx> <tblocky> <tblockz> <face> <data_type> <async> <autoconfig>\n");
		return (1);
	}
	ni = atoi(argv[1]);
	nj = atoi(argv[2]);
	nk = atoi(argv[3]);

	tx = atoi(argv[4]);
	ty = atoi(argv[5]);
	tz = atoi(argv[6]);

	face_id = atoi(argv[7]);

	l_type = atoi(argv[8]);
	set_type((meta_type_id) l_type);

	async = (a_bool) atoi(argv[9]);
	autoconfig = (a_bool) atoi(argv[10]);

	dimblock_red[0] = tx, dimblock_red[1] = ty, dimblock_red[2] = tz;
	dimgrid_red[0] = ni / tx + ((ni % tx) ? 1 : 0);
	dimgrid_red[1] = nj / ty + ((nj % ty) ? 1 : 0);
	dimgrid_red[2] = nk / tz + ((nk % tz) ? 1 : 0);
	dimarray_3d[0] = ni, dimarray_3d[1] = nj, dimarray_3d[2] = nk;

	//These are for the library reduction, which we use for sums
	void * sum_gpu, *zero;
	sum_gpu = malloc(g_typesize);
	zero = malloc(g_typesize);
	switch (g_type) {
	case a_db:
		*(double*) zero = 0;
		break;

	case a_fl:
		*(float*) zero = 0;
		break;

	case a_ul:
		*(unsigned long*) zero = 0;
		break;

	case a_in:
		*(int *) zero = 0;
		break;

	case a_ui:
		*(unsigned int *) zero = 0;
		break;
	}

#ifdef WITH_TIMERS
	metaTimersInit();
#endif

	gpu_initialize(rank);
	data_allocate(ni, nj, nk);
	data_initialize(ni, nj, nk);

	MPI_Request request;
	MPI_Status status;
	for (i = 0; i < 1; i++) {

		if (rank == 0) {
			//Only process 0 actually needs to initialize data
			// proc1 is just a relay that tests things on the receiving
			// end and mirror's the data back
			//copy the unmodified prism to device
			meta_copy_h2d(dev_data3, data3, ni * nj * nk * g_typesize, async);
			//check_buffer(data3, dev_data3, ni*nj*nk);
			//Validate grid and block sizes (if too big, shrink the z-dim and add iterations)
			for (;
					meta_validate_worksize(&dimgrid_red, &dimblock_red) != 0
							&& dimblock_red[2] > 1;
					dimgrid_red[2] <<= 1, dimblock_red[2] >>= 1)
				;
			//zero out the reduction sum
			meta_copy_h2d(reduction, zero, g_typesize, async);
			//reduce the face to check that the transfer was correct
			//accurately sets start and end indices to sum each face
			arr_start[0] = ((face_id == 3) ? ni - 1 : 0);
			arr_end[0] = ((face_id == 2) ? 0 : ni - 1);
			arr_start[1] = ((face_id == 5) ? nj - 1 : 0);
			arr_end[1] = ((face_id == 4) ? 0 : nj - 1);
			arr_start[2] = ((face_id == 1) ? nk - 1 : 0);
			arr_end[2] = ((face_id == 0) ? 0 : nk - 1);
			//check_dims(dimarray_3d, arr_start, arr_end);
//		printf("Integrity check dim(%d, %d, %d) start(%d, %d, %d) end(%d, %d, %d)\n", dimarray_3d[0], dimarray_3d[1], dimarray_3d[2], arr_start[0], arr_start[1], arr_start[2], arr_end[0], arr_end[1], arr_end[2]);
			a_err ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, dev_data3, &dimarray_3d,
					&arr_start, &arr_end, reduction, g_type, async);
			//a_dim3 testgrid, testblock;
			//testgrid[0] = testgrid[1] = testgrid[2] = 1;
			//testblock[0] = 16;
			//testblock[1] = 8;
			//testblock[2] = 1;
			//a_err ret = meta_reduce(&testgrid, &testblock, dev_data3, &dimarray_3d, &arr_start, &arr_end, reduction, g_type, async);
			printf("Reduce Error: %d\n", ret);
			//pull the sum back
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("Initial Face Integrity Check: %s\n",
					check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
							(face_id < 2 || face_id > 3 ? ni : nk),
							(face_id & 1 ?
									(face_id < 2 ?
											nk - 1 :
											(face_id < 4 ? ni - 1 : nj - 1)) :
									0)) ? "FAILED" : "PASSED");

			//pack the face
			//TODO set a_dim3 structs once the internal implementation respects them
			face_spec = make_face(face_id, ni, nj, nk);
			ret = meta_pack_face(NULL, NULL, dev_face[face_id], dev_data3,
					face_spec, g_type, async);
			printf("Pack Return Val: %d\n", ret);
			//check_buffer(face[face_id], dev_face[face_id], face_spec->size[0]*face_spec->size[1]*face_spec->size[2]);

			//reduce the packed face to check that packing was correct
			meta_copy_h2d(reduction, zero, g_typesize, async);
			dim_array2d[0] = face_spec->size[2], dim_array2d[1] =
					face_spec->size[1], dim_array2d[2] = face_spec->size[0];
			start_2d[0] = start_2d[1] = start_2d[2] = 0;
			end_2d[0] = (dim_array2d[0] == 1 ? 0 : ni - 1);
			end_2d[1] = (dim_array2d[1] == 1 ? 0 : nj - 1);
			end_2d[2] = (dim_array2d[2] == 1 ? 0 : nk - 1);

			//check_dims(dim_array2d, start_2d, end_2d);
			ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, dev_face[face_id],
					&dim_array2d, &start_2d, &end_2d, reduction, g_type, async);
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("Packed Face Integrity Check: %s\n",
					check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
							(face_id < 2 || face_id > 3 ? ni : nk),
							(face_id & 1 ?
									(face_id < 2 ?
											nk - 1 :
											(face_id < 4 ? ni - 1 : nj - 1)) :
									0)) ? "FAILED" : "PASSED");

			//transpose the packed face (into the companion face's unoccupied buffer)
			trans_dim[0] = (
					face_spec->size[2] == 1 ?
							face_spec->size[1] : face_spec->size[2]);
			trans_dim[1] = (
					face_spec->size[0] == 1 ?
							face_spec->size[1] : face_spec->size[0]);
			trans_dim[2] = 1;
			rtrans_dim[0] = trans_dim[1];
			rtrans_dim[1] = trans_dim[0];
			rtrans_dim[2] = 1;

			void * stuff = calloc(
					face_spec->size[0] * face_spec->size[1]
							* face_spec->size[2], g_typesize);
			meta_copy_h2d(dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
					stuff,
					g_typesize * face_spec->size[0] * face_spec->size[1]
							* face_spec->size[2], async);
			//printf("**BEFORE**\n");
			//check_buffer(face[face_id], dev_face[(face_id & 1) ? face_id-1 : face_id+1], face_spec->size[0]*face_spec->size[1]*face_spec->size[2]);
			//printf("**********\n");
			//check_dims(dimgrid_red, dimblock_red, trans_dim);
			//TODO Figure out what's wrong with transpose and re-enable
			ret = meta_transpose_face(NULL, NULL, dev_face[face_id],
					dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
					&trans_dim, &trans_dim, g_type, async);
			printf("Transpose error: %d\n", ret);
			//printf("**AFTER***\n");
			//check_buffer(face[face_id], dev_face[(face_id & 1) ? face_id-1 : face_id+1], face_spec->size[0]*face_spec->size[1]*face_spec->size[2]);
			//printf("**********\n");

			//reduce the specific sums needed to check that transpose was correct
			meta_copy_h2d(reduction, zero, g_typesize, async);
			//shuffle the (local) X/Y dimension
			rtrans_dim[0] = trans_dim[1];
			rtrans_dim[1] = trans_dim[0];
			rtrans_dim[2] = 1;
			start_2d[0] = start_2d[1] = start_2d[2] = 0;
			end_2d[0] = trans_dim[0] - 1, end_2d[1] = trans_dim[1] - 1, end_2d[2] =
					0;
			//check_dims(rtrans_dim, start_2d, end_2d);
			ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red,
					dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
					&trans_dim, &start_2d, &end_2d, reduction, g_type, async);
			//ret = meta_reduce(NULL, NULL, dev_face[(face_id & 1)? face_id-1 : face_id+1], &trans_dim, &start_2d, &end_2d, reduction, g_type, async);
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("Transposed Face Integrity Check: %s\n",
					check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
							(face_id < 2 || face_id > 3 ? ni : nk),
							(face_id & 1 ?
									(face_id < 2 ?
											nk - 1 :
											(face_id < 4 ? ni - 1 : nj - 1)) :
									0)) ? "FAILED" : "PASSED");

			//transpose the face back
			//TODO figure out what's wrong with transpose and re-enable
			ret = meta_transpose_face(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red,
					dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
					dev_face[face_id], &rtrans_dim, &rtrans_dim, g_type, async);
			//check_buffer(face[face_id], dev_face[face_id], face_spec->size[0]*face_spec->size[1]*face_spec->size[2]);
			//reduce the specified sums to ensure the reverse transpose worked too
			meta_copy_h2d(reduction, zero, g_typesize, async);
			start_2d[0] = start_2d[1] = start_2d[2] = 0;
			end_2d[0] = rtrans_dim[0] - 1, end_2d[1] = rtrans_dim[1] - 1, end_2d[2] =
					0;
			dimgrid_tr_red[0] = dimgrid_red[1];
			dimgrid_tr_red[1] = dimgrid_red[0];
			dimgrid_tr_red[2] = dimgrid_red[2];
			ret = meta_reduce(autoconfig ? NULL : &dimgrid_tr_red,
					autoconfig ? NULL : &dimblock_red,
					dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
					&rtrans_dim, &start_2d, &end_2d, reduction, g_type, async);
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("Retransposed Face Integrity Check: %s\n",
					check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
							(face_id < 2 || face_id > 3 ? ni : nk),
							(face_id & 1 ?
									(face_id < 2 ?
											nk - 1 :
											(face_id < 4 ? ni - 1 : nj - 1)) :
									0)) ? "FAILED" : "PASSED");
			;

			//send the packed face to proc1
			ret = meta_mpi_packed_face_send(1, dev_face[face_id],
					trans_dim[0] * trans_dim[1], i, &request, g_type, async);

//Force the recv and unpack to finish
			meta_flush();
//At this point the send should be forced to complete
// which means there's either a failure in the SP helper or the RP helper

			//receive and unpack the face
			//TODO set a_dim3 structs - i believe these are fine
			//TODO set the face_spec - believe these are fine
			ret = meta_mpi_recv_and_unpack_face(
					autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, 1, face_spec, dev_data3,
					dev_face[face_id], i, &request, g_type, async);

//Force the recv and unpack to finish
			meta_flush();
			meta_copy_h2d(reduction, zero, g_typesize, async);
			arr_start[0] = ((face_id == 3) ? ni - 1 : 0);
			arr_end[0] = ((face_id == 2) ? 0 : ni - 1);
			arr_start[1] = ((face_id == 5) ? nj - 1 : 0);
			arr_end[1] = ((face_id == 4) ? 0 : nj - 1);
			arr_start[2] = ((face_id == 1) ? nk - 1 : 0);
			arr_end[2] = ((face_id == 0) ? 0 : nk - 1);
			//check_dims(dimarray_3d, arr_start, arr_end);
			ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, dev_data3, &dimarray_3d,
					&arr_start, &arr_end, reduction, g_type, async);
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("RecvAndUnpacked ZeroFace Integrity Check: %s\n",
					(check_fp(0.0, *((double*) sum_gpu), 0.000001)) ?
							"PASSED" : "FAILED");
			if (!check_fp(0.0, *((double*) sum_gpu), 0.000001))
				printf("\tExpected [0.0], returned [%f]!\n", sum_gpu);

			ret = meta_mpi_recv_and_unpack_face(
					autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, 1, face_spec, dev_data3,
					dev_face[face_id], i, &request, g_type, async);

//Force the recv and unpack to finish
			meta_flush();
			meta_copy_h2d(reduction, zero, g_typesize, async);
			arr_start[0] = ((face_id == 3) ? ni - 1 : 0);
			arr_end[0] = ((face_id == 2) ? 0 : ni - 1);
			arr_start[1] = ((face_id == 5) ? nj - 1 : 0);
			arr_end[1] = ((face_id == 4) ? 0 : nj - 1);
			arr_start[2] = ((face_id == 1) ? nk - 1 : 0);
			arr_end[2] = ((face_id == 0) ? 0 : nk - 1);
			//check_dims(dimarray_3d, arr_start, arr_end);
			ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, dev_data3, &dimarray_3d,
					&arr_start, &arr_end, reduction, g_type, async);
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("RecvAndUnpacked Face Integrity Check: %s\n",
					check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
							(face_id < 2 || face_id > 3 ? ni : nk),
							(face_id & 1 ?
									(face_id < 2 ?
											nk - 1 :
											(face_id < 4 ? ni - 1 : nj - 1)) :
									0)) ? "FAILED" : "PASSED");
			//TODO reduce the specified sub-sums on the face to further check accuracy of placement
		} else {
			//receive the packed buf
			//TODO fill in <buf_leng>
			face_spec = make_face(face_id, ni, nj, nk);
			meta_face * opp_face = make_face(
					(face_id & 1 ? face_id - 1 : face_id + 1), ni, nj, nk);
			//one of the faces should always be size = 1, since we're only taking a 1-deep subsection
			// thus, the size of the recv buffer can be the product of all 3 elements
			a_err ret = meta_mpi_packed_face_recv(0, dev_face[face_id],
					face_spec->size[0] * face_spec->size[1]
							* face_spec->size[2], i, &request, g_type, async);

//Force the recv and unpack to finish
//FIXME this flush appears to invalidate something needed by the ensuing reduce at L:570
			meta_flush();
			//check_buffer(face[face_id], dev_face[face_id], face_spec->size[0]*face_spec->size[1]*face_spec->size[2]);i
			//check_buffer(face[face_id], dev_face[face_id], face_spec->size[0]*face_spec->size[1]*face_spec->size[2]);

			//TODO reduce the packed buf to check the sum - this should be fine
			meta_copy_h2d(reduction, zero, g_typesize, async);
			start_2d[0] = start_2d[1] = start_2d[2] = 0;
			end_2d[0] = face_spec->size[2] - 1, end_2d[1] = face_spec->size[1]
					- 1, end_2d[2] = face_spec->size[0] - 1;
			//FIXME: Make sure this reduce doesn't care if x or y is length 1, it *should* be fine
			trans_dim[0] = face_spec->size[2], trans_dim[1] =
					face_spec->size[1], trans_dim[2] = face_spec->size[0];
			//check_dims(trans_dim, start_2d, end_2d);
			ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, dev_face[face_id],
					&trans_dim, &start_2d, &end_2d, reduction, g_type, async);
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("Received Face Integrity Check: %s\n",
					check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
							(face_id < 2 || face_id > 3 ? ni : nk),
							(face_id & 1 ?
									(face_id < 2 ?
											nk - 1 :
											(face_id < 4 ? ni - 1 : nj - 1)) :
									0)) ? "FAILED" : "PASSED");

			//unpack it, reduce/test the sum again
			//TODO set a_dim3 structs - these should be fine until the lib respects user provided ones
			ret = meta_unpack_face(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, dev_face[face_id],
					dev_data3, face_spec, g_type, async);
			printf("Unpack retval: %d\n", ret);

//Force the recv and unpack to finish
			meta_flush();
			//check_buffer(data3, dev_data3, ni*nj*nk);
			//TODO reduce the unpacked face to test the sum(s)
			meta_copy_h2d(reduction, zero, g_typesize, async);
			arr_start[0] = ((face_id == 3) ? ni - 1 : 0);
			arr_end[0] = ((face_id == 2) ? 0 : ni - 1);
			arr_start[1] = ((face_id == 5) ? nj - 1 : 0);
			arr_end[1] = ((face_id == 4) ? 0 : nj - 1);
			arr_start[2] = ((face_id == 1) ? nk - 1 : 0);
			arr_end[2] = ((face_id == 0) ? 0 : nk - 1);
			//check_dims(dimarray_3d, arr_start, arr_end);
			ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, dev_data3, &dimarray_3d,
					&arr_start, &arr_end, reduction, g_type, async);
			meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
			//The 4 ternaries ensure the right args are passed to match the face
			// so this one call will work for any face
			printf("Unpacked Face Integrity Check: %s\n",
					check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
							(face_id < 2 || face_id > 3 ? ni : nk),
							(face_id & 1 ?
									(face_id < 2 ?
											nk - 1 :
											(face_id < 4 ? ni - 1 : nj - 1)) :
									0)) ? "FAILED" : "PASSED");
			//combined pack and send it back
			//TODO set the face_spec
			//Send a face of zeroes
			ret = meta_mpi_pack_and_send_face(autoconfig ? NULL : &dimgrid_red,
					&autoconfig ? NULL : &dimblock_red, 0, opp_face, dev_data3,
					dev_face[face_id], i, &request, g_type, async);

//Force the recv and unpack to finish
			meta_flush();
			//Then send a real face
			ret = meta_mpi_pack_and_send_face(autoconfig ? NULL : &dimgrid_red,
					autoconfig ? NULL : &dimblock_red, 0, face_spec, dev_data3,
					dev_face[face_id], i, &request, g_type, async);

//Force the recv and unpack to finish
			meta_flush();
		}
	}
	//deallocate_();
#ifdef WITH_TIMERS
	metaTimersFinish();
#endif //WITH_TIMERS
	meta_mpi_finalize();
}
