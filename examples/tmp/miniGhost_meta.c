#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
	#include <metamorph.h>

#define DEBUG2
#define INTEROP
#define MAX_NUM_NEIGHBORS	6
//Should be set on the compile line, but if not, default to double
#ifdef DOUBLE 
	#define G_TYPE double
	#define CL_G_TYPE cl_double
#elif defined(FLOAT)
	#define G_TYPE float
	#define CL_G_TYPE cl_float
#else
	#define G_TYPE double
	#define CL_G_TYPE cl_double
#endif

// host buffers
void *data3, *data3_op;
int ni, nj, nk, nm;
int nneighs = 2; //number of neighbors NUM_NEIGHS
int neighs[MAX_NUM_NEIGHBORS] = {};

G_TYPE r_val;
G_TYPE global_sum;
	a_err err;
	void * dev_d3, *dev_d3_op,  *result, *dev_sendbuf, *dev_recvbuf;
	meta_dim3 grid, block, array, a_start, a_end, a_start2, a_end2;
void init(int rank, int comm_size) {
	#ifdef WITH_CUDA
	if (rank == 0) meta_set_acc(0, metaModePreferCUDA);
	else fprintf(stderr, "WITH_CUDA should only be defined for rank 0, I have rank [%d]\n", rank);
	#elif defined(WITH_OPENCL)
		#ifdef DEBUG2
		if (rank ==0) meta_set_acc(-1, metaModePreferOpenCL);
		#else
		if (rank ==2) meta_set_acc(0, metaModePreferOpenCL);
		#endif
	else fprintf(stderr, "WITH_OPENCL should only be defined for rank 2, I have rank [%d]\n", rank);
	#elif defined(WITH_OPENMP)
		#ifdef DEBUG2
		if (rank ==0) meta_set_acc(0, metaModePreferOpenMP);
		#else
		if (rank == 1 || rank == 3) meta_set_acc(0, metaModePreferOpenMP);
		#endif
	else fprintf(stderr, "WITH_OPENMP should only be defined for ranks 1 and 3, I have rank [%d]\n", rank);
	#else
		#error MetaMorph needs either WITH_CUDA, WITH_OPENCL, or WITH_OPENMP
	#endif

		// default processor grid (comm_size, 1, 1)
		// if rank != 0 then my west neighbour is rank-1
		//if rank != comm_size-1 then my east neighbor is rank+1

		//for interoperability test, we communicate with east and west,
		// even if they don't exist (i.e. torus)

		// support SVAF communication
}

void cleanup() {}

void data_allocate() {
	data3 = malloc(sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2));
	data3_op = malloc(sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2));
	if (err = meta_alloc(&dev_d3, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2))) fprintf(stderr, "ERROR allocating dev_d3: [%d]\n", err);
	if (err = meta_alloc(&dev_d3_op, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2))) fprintf(stderr, "ERROR allocating dev_d3: [%d]\n", err);
	if (err = meta_alloc(&result, sizeof(G_TYPE))) fprintf(stderr, "ERROR allocating result: [%d]\n", err);
	if (err = meta_alloc(&dev_sendbuf, sizeof(G_TYPE)*(nj+2)*(nk+2))) fprintf(stderr, "Error allocating dev_sendbuf: [%d]\n", err);
	if (err = meta_alloc(&dev_recvbuf, sizeof(G_TYPE)*(nj+2)*(nk+2))) fprintf(stderr, "Error allocating dev_recvbuf: [%d]\n", err);
} 

void data_initialize(int rank) {
	G_TYPE * l_data3 = (G_TYPE *) data3;
	G_TYPE * l_data3_op = (G_TYPE *) data3_op;

	int iter;
	int i, j, k;
	for (i = ni+1; i >= 0; i--) {
		for (j = nj+1; j >= 0; j--) {
			for (k = nk+1; k >= 0; k--) {
				if (i == 0 || j == 0 || k == 0 || i == ni+1 || j == nj+1 || k == nk+1) {
					l_data3[i+j*(ni+2)+k*(ni+2)*(nj+2)] = 0.0f;
				} else {
					l_data3[i+j*(ni+2)+k*(ni+2)*(nj+2)] = i+j+k + (ni*rank);
				}
			}
		}
	}

	for (i = ni+1; i >= 0; i--) {
		for (j = nj+1; j >= 0; j--) {
			for (k = nk+1; k >= 0; k--) {
				l_data3_op[i+j*(ni+2)+k*(ni+2)*(nj+2)] = 0.0f;
			}
		}
	}
}

void deallocate() {
	free(data3);
	free(data3_op);
	meta_free(dev_d3);
	meta_free(dev_d3_op);
	meta_free(result);
	meta_free(dev_sendbuf);
	meta_free(dev_recvbuf);
}

meta_face * make_slab2d_from_3d(int face, int ni, int nj, int nk, int thickness) {
	meta_face * ret = (meta_face*)malloc(sizeof(meta_face));
	ret->count = 3;
	ret->size = (int*)malloc(sizeof(int)*3);
	ret->stride = (int*)malloc(sizeof(int)*3);
	//all even faces start at the origin, all others start at some offset
	// defined by the dimensions of the prism
	/* 0 -> N, 1 -> S, 2 -> E, 3-> W, 4 -> T, 5 -> B*/
	if (face & 1) {
		if (face == 1) ret->start = ni*nj*(nk-thickness);
		if (face == 3) ret->start = ni-thickness;
		if (face == 5) ret->start = ni*(nj-thickness);
	} else ret->start = 0;
	ret->size[0] = nk, ret->size[1] = nj, ret->size[2] = ni;
	if (face < 2) ret->size[0] = thickness;
	if (face > 3) ret->size[1] = thickness;
	if (face > 1 && face < 4) ret->size[2] = thickness;
	ret->stride[0] = ni*nj, ret->stride[1] = ni, ret->stride[2] = 1;
	printf("Generated Face:\n\tcount: %d\n\tstart: %d\n\tsize: %d %d %d\n\tstride: %d %d %d\n", ret->count, ret->start, ret->size[0], ret->size[1], ret->size[2], ret->stride[0], ret->stride[1], ret->stride[2]);
	return ret;
}

#ifdef DEBUG
void print_grid(G_TYPE * grid) {
	int i,j,k;
	for (k = 0; k < nk; k++) {
		for (j =0; j < nj; j++) {
			for (i=0; i < ni+1; i++) {
				printf("[%f] ", grid[i+ j*(ni+1) + k*nj*(ni+1)]);
			}
			printf("\n");
		}
		printf("\n");
	}	
}
#endif


int main(int argc, char **argv) {
	meta_mpi_init(&argc, &argv);

	int comm_sz;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Request request;

	int niters, async, autoconfig, scaling;
	if (argc < 9) {
		fprintf(stderr, "<ni> <nj> <nk> <nm> <tsteps> <async> <autoconfig> <scaling>\n");
		return (-1);
	}
	ni = atoi(argv[1]);
	nj = atoi(argv[2]);
	nk = atoi(argv[3]);
	nm = atoi(argv[4]); //num of variables
	niters = atoi(argv[5]);
	async = atoi(argv[6]);
	autoconfig = atoi(argv[7]);
	scaling = atoi(argv[8]); //0 weak , 1 strong

	printf("<%d> <%d> <%d> <%d> <%d> <%d>\n", ni, nj, nk, niters, async, autoconfig);

	init(rank, comm_sz);
	metaTimersInit();
	data_allocate();
	data_initialize(rank);
#ifdef DEBUG
	print_grid(data3);
#endif
	int ct;
	struct timeval start, end;
	void *temp;
	G_TYPE zero = 0.0;
	grid[0] = ni/4, grid[1] = nj/4, grid[2] = nk/2; //Assume powers of 2, for simplicity
	block[0] = 4, block[1] = 4, block[2] = 2; //Hardcoded to good values
	array[0] = ni+2, array[1] = nj+2, array[2] = nk+2;
	a_start[0] = a_start[1] = a_start[2] = 0;
	a_end[0] = ni+1, a_end[1] = nj+1, a_end[2] = nk+1;
	a_start2[0] = a_start2[1] = a_start2[2] = 1;
	a_end2[0] = ni-1, a_end2[1] = nj-1, a_end2[2] = nk-1;

	meta_face * send_face, * recv_face;
	send_face = make_slab2d_from_3d(3, ni+2, nj+2, nk+2, 1); //send west
	recv_face = make_slab2d_from_3d(2, ni+2, nj+2, nk+2, 1); //rec east

#ifdef WITH_OPENMP
	//do a throwaway pack directly via the OpenMP core to get threads spawned and keep that overhead
	// out of the timing loop
	int *remain_dim = (int*)malloc(sizeof(int)*send_face->count);
	int xi;
	remain_dim[send_face->count-1] = 1;
	for (xi = send_face->count-2; xi >= 0; xi--) remain_dim[xi] = remain_dim[xi+1]*send_face->size[xi+1];
	omp_pack_face_kernel_db(dev_sendbuf, dev_d3, send_face, remain_dim);
	free(remain_dim);
#endif

	if (err = meta_copy_h2d(dev_d3, data3, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2), false)) fprintf(stderr, "ERROR Init dev_d3 failed: [%d]\n", err);
	if (err = meta_copy_h2d(dev_d3_op, data3, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2), false)) fprintf(stderr, "ERROR Init dev_d3 failed: [%d]\n", err);

#ifdef DEBUG
	printf("Post-H2D grid");
	meta_copy_d2h(data3, dev_d3, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2), false);
	print_grid(data3);
#endif

	gettimeofday(&start, NULL);
	/* Jacobi Solver */
	for (ct = niters-1; ct >= 0; ct--) {
	#if defined(DOUBLE)
		#ifdef INTEROP
		//set up async recv and unpack
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+comm_sz-1)%comm_sz, recv_face, dev_d3, dev_recvbuf, ct, &request, meta_db, 1);
		//pack and send
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+1)%comm_sz, send_face, dev_d3, dev_sendbuf, ct, &request, meta_db, 0);
		#else
		if(rank !=0 ) //my west neighbour is rank-1
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank-1), recv_face, dev_d3, dev_recvbuf, ct, &request, meta_db, 1);

		if(rank != comm_size-1 ) //my east neighbor is rank+1
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, rank+1, send_face, dev_d3, dev_sendbuf, ct, &request, meta_db, 0);
		#endif
	#elif defined(FLOAT)
		#ifdef INTEROP
		//set up async recv and unpack
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+comm_sz-1)%comm_sz, recv_face, dev_d3, dev_recvbuf, ct, &request, meta_fl, 1);
		//pack and send
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+1)%comm_sz, send_face, dev_d3, dev_sendbuf, ct, &request, meta_fl, 0);
		#else
		if(rank !=0 ) //my west neighbour is rank-1
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+comm_sz-1)%comm_sz, recv_face, dev_d3, dev_recvbuf, ct, &request, meta_fl, 1);
		if(rank != comm_size-1 ) //my east neighbor is rank+1
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+1)%comm_sz, send_face, dev_d3, dev_sendbuf, ct, &request, meta_fl, 0);
		#endif
	#else
		#error Unsupported G_TYPE, must be double or float
	#endif
		// flush
		meta_flush();

#ifdef DEBUG
	printf("Pre-stencil grid");
	meta_copy_d2h(data3, dev_d3, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2), false);
	print_grid(data3);
#endif	

	#if defined(DOUBLE)
	    meta_stencil_3d7p(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, dev_d3_op, &array, &a_start, &a_end, meta_db, false);
	    meta_copy_d2d(dev_d3, dev_d3_op, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2), false);
	    meta_copy_h2d(result, &zero, sizeof(G_TYPE), true);
		meta_reduce(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, &array, &a_start2, &a_end2, result, meta_db, true);
		meta_copy_d2h(&r_val, result, sizeof(G_TYPE), false);
		//mpi reduce
		MPI_Reduce(&r_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	#elif defined(FLOAT)
	    meta_stencil_3d7p(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, dev_d3_op, &array, &a_start, &a_end, meta_fl, false);
	    meta_copy_d2d(dev_d3, dev_d3_op, sizeof(G_TYPE)*(ni+2)*(nj+2)*(nk+2), false);
	    meta_copy_h2d(result, &zero, sizeof(G_TYPE), true);
		meta_reduce(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, &array, &a_start2, &a_end2, result, meta_fl, true);
		meta_copy_d2h(&r_val, result, sizeof(G_TYPE), true);
		//mpi reduce
		MPI_Reduce(&r_val, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	#else
		#error Unsupported G_TYPE, must be double or float
	#endif
		//if (rank == 0 && global_sum != check) fprintf(stderr, "Error, computed dot-product invalid!\n\tExpect: [%f]\tReturn: [%f]\n", check, r_val);
		//temp = dev_d3;
		//dev_d3 = dev_d3_op;
		//dev_d3_op = dev_d3;
	}
	gettimeofday(&end, NULL);
	double time = (end.tv_sec-start.tv_sec)*1000000.0 + (end.tv_usec-start.tv_usec);
	printf("Local partial sum on rank[%d]: [%f]\n", rank, r_val);
	printf("[%d] Tests completed on rank[%d] with matrices of size[%d][%d][%d]\n\t[%f] Total time (us)\n\t[%f] Average time/iteration (us)\n\t[%f] Average time/element (us)\n", niters, rank, ni, nj, nk, time, time/niters, time/niters/(ni*nj*nk));
	deallocate();

	metaTimersFinish();
	cleanup();

	meta_mpi_finalize();
	return 0;
}
