#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <metamorph.h>

#define DEBUG2

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

//Sum of a region when all cells are filled by i+j+k (i=0:ni-1, j=0:nj-1, k=0:nk-1)
#define SUM_REG(a,b,c) ((((a)*(b)*(c))/2.0)*((a)+(b)+(c)-3))

// host buffers
void *data3, *data3_2;

// dev buffers
void * dev_d3, *dev_d32, *result, *dev_sendbuf, *dev_recvbuf;

int ni, nj, nk;
G_TYPE r_val;
G_TYPE global_sum;
a_err err;
a_dim3 grid, block, array, a_start, a_end;


void init(int rank) {
#ifdef WITH_CUDA
	if (rank == 0) choose_accel(0, metaModePreferCUDA);
	else fprintf(stderr, "WITH_CUDA should only be defined for rank 0, I have rank [%d]\n", rank);
#elif defined(WITH_OPENCL)
#ifdef DEBUG2
	if (rank ==0) choose_accel(-1, metaModePreferOpenCL);
#else
	if (rank ==2) choose_accel(0, metaModePreferOpenCL);
#endif
	else fprintf(stderr, "WITH_OPENCL should only be defined for rank 2, I have rank [%d]\n", rank);
#elif defined(WITH_OPENMP)
#ifdef DEBUG2
	if (rank ==0) choose_accel(0, metaModePreferOpenMP);
#else
	if (rank == 1 || rank == 3) choose_accel(0, metaModePreferOpenMP);
#endif
	else fprintf(stderr, "WITH_OPENMP should only be defined for ranks 1 and 3, I have rank [%d]\n", rank);
#else
#error MetaMorph needs either WITH_CUDA, WITH_OPENCL, or WITH_OPENMP
#endif
}

void cleanup() {
}

void data_allocate() {
	data3 = malloc(sizeof(G_TYPE) * (ni + 1) * nj * nk);
	data3_2 = malloc(sizeof(G_TYPE) * (ni + 1) * nj * nk);
	if (err = meta_alloc(&dev_d3, sizeof(G_TYPE) * (ni + 1) * nj * nk))
		fprintf(stderr, "ERROR allocating dev_d3: [%d]\n", err);
	if (err = meta_alloc(&dev_d32, sizeof(G_TYPE) * (ni + 1) * nj * nk))
		fprintf(stderr, "ERROR allocating dev_d3: [%d]\n", err);
	if (err = meta_alloc(&result, sizeof(G_TYPE)))
		fprintf(stderr, "ERROR allocating result: [%d]\n", err);
	if (err = meta_alloc(&dev_sendbuf, sizeof(G_TYPE) * nj * nk))
		fprintf(stderr, "Error allocating dev_sendbuf: [%d]\n", err);
	if (err = meta_alloc(&dev_recvbuf, sizeof(G_TYPE) * nj * nk))
		fprintf(stderr, "Error allocating dev_recvbuf: [%d]\n", err);
}

void data_initialize(int rank) {
	G_TYPE * l_data3 = (G_TYPE *) data3;
	G_TYPE * l_data3_2 = (G_TYPE *) data3_2;

	int iter;
	int i, j, k;

#if DEBUG2
	for (i = ni; i >= 0; i--) {
		for (j = nj - 1; j >= 0; j--) {
			for (k = nk - 1; k >= 0; k--) {
				l_data3_2[i + j * (ni + 1) + k * (ni + 1) * nj] = 1.0f;
			}
		}
	}
#else
	for (i = ni; i >= 0; i--) {
		for (j = nj - 1; j >= 0; j--) {
			for (k = nk - 1; k >= 0; k--) {
				if (i == 0) {
					l_data3[i + j * (ni + 1) + k * (ni + 1) * nj] = 0.0f;
				} else {
					l_data3[i + j * (ni + 1) + k * (ni + 1) * nj] = i + j + k
							+ (ni * rank);
				}
			}
		}
	}
#endif
}

void deallocate() {
	free(data3);
	free(data3_2);
	meta_free(dev_d3);
	meta_free(dev_d32);
	meta_free(result);
	meta_free(dev_sendbuf);
	meta_free(dev_recvbuf);
}

meta_face * make_slab2d_from_3d(int face, int ni, int nj, int nk, int thickness) {
	meta_face * ret = (meta_face*) malloc(sizeof(meta_face));
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
	//(MPI+OMP)
	// MM: MPI-Init & finalize -> 7 (7)
	meta_mpi_init(&argc, &argv);
	int comm_sz;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Request request;

	int niters, async, autoconfig;
	if (argc < 7) {
		fprintf(stderr, "<ni> <nj> <nk> <iterations> <async> <autoconfig>\n");
		return (-1);
	}
	ni = atoi(argv[1]);
	nj = atoi(argv[2]);
	nk = atoi(argv[3]);
	niters = atoi(argv[4]);
	async = atoi(argv[5]);
	autoconfig = atoi(argv[6]);

	printf("<%d> <%d> <%d> <%d> <%d> <%d>\n", ni, nj, nk, niters, async,
			autoconfig);

	init(rank); // MM:Context-Init -> 3
	metaTimersInit(); //MM: Timers-Init & finalize ->2 (ignore)
	data_allocate(); //MM: alloc and free -> 10
	data_initialize(rank);
#ifdef DEBUG
	print_grid((G_TYPE * ) data3);
#endif
	//(The zero face is actually the max face from the far process
	// so compute the sum including the max face, and subtract off the min face that doesn't exist
	G_TYPE check = SUM_REG((ni*comm_sz+1),nj,nk) - SUM_REG(1, nj, nk);
	int ct, flag = 0;
	struct timeval start, end;
	G_TYPE zero = 0.0;
	//grid[0] = ni/32, grid[1] = nj/4, grid[2] = nk/2; //Assume powers of 2, for simplicity
	//block[0] = 32, block[1] = 4, block[2] = 2; //Hardcoded to good values
	grid[0] = ni / 4, grid[1] = nj / 4, grid[2] = nk / 2; //Assume powers of 2, for simplicity
	block[0] = 4, block[1] = 4, block[2] = 2; //Hardcoded to good values
	array[0] = ni + 1, array[1] = nj, array[2] = nk;
	a_start[0] = a_start[1] = a_start[2] = 0;
	a_end[0] = ni - 1, a_end[1] = nj - 1, a_end[2] = nk - 1;

	// MM: Face specs -> 3 (3)
	meta_face * send_face, *recv_face;
	send_face = make_slab2d_from_3d(3, ni + 1, nj, nk, 1);
	recv_face = make_slab2d_from_3d(2, ni + 1, nj, nk, 1);

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

	//MM: Data-copy -> 2
	if (err = meta_copy_h2d(dev_d3, data3, sizeof(G_TYPE) * (ni + 1) * nj * nk,
			false))
		fprintf(stderr, "ERROR Init dev_d3 failed: [%d]\n", err);
	if (err = meta_copy_h2d(dev_d32, data3_2,
			sizeof(G_TYPE) * (ni + 1) * nj * nk, false))
		fprintf(stderr, "ERROR Init dev_d3 failed: [%d]\n", err);

#ifdef DEBUG
	printf("Post-H2D grid");
	meta_copy_d2h(data3, dev_d3, sizeof(G_TYPE)*(ni+1)*nj*nk, false);
	print_grid(data3);
#endif

	gettimeofday(&start, NULL);
	for (ct = niters - 1; ct > -1; ct--) {

		//MM: data marshaling -> 3 (8)
#if defined(DOUBLE)
		//set up async recv and unpack
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+comm_sz-1)%comm_sz, recv_face, dev_d3, dev_recvbuf, ct, &request, a_db, 1);
		//pack and send
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+1)%comm_sz, send_face, dev_d3, dev_sendbuf, ct, &request, a_db, 0);
#elif defined(FLOAT)
		//set up async recv and unpack
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+comm_sz-1)%comm_sz, recv_face, dev_d3, dev_recvbuf, ct, &request, a_fl, 1);
		//pack and send
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+1)%comm_sz, send_face, dev_d3, dev_sendbuf, ct, &request, a_fl, 0);
#else
#error Unsupported G_TYPE, must be double or float
#endif
		// flush
		meta_flush();
		//local reduction
#ifdef DEBUG
		printf("Pre-reduce grid");
		if (err = meta_alloc(&result, sizeof(G_TYPE))) fprintf(stderr, "ERROR allocating result: [%d]\n", err);
		meta_copy_d2h(data3, dev_d3, sizeof(G_TYPE)*(ni+1)*nj*nk, false);
		print_grid(data3);
#endif	

		//MM: dotP+Data copy -> 4 (5)
#if defined(DOUBLE)
		meta_copy_h2d(result, &zero, sizeof(G_TYPE), true);
		meta_dotProd(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, dev_d32, &array, &a_start, &a_end, result, a_db, true);
		//meta_reduce(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, &array, &a_start, &a_end, result, a_db, true);
		meta_copy_d2h(&r_val, result, sizeof(G_TYPE), false);
		//mpi reduce
		MPI_Reduce(&r_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#elif defined(FLOAT)
		meta_copy_h2d(result, &zero, sizeof(G_TYPE), true);
		meta_dotProd(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, dev_d32, &array, &a_start, &a_end, result, a_fl, true);
		//meta_reduce(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, dev_d3, &array, &a_start, &a_end, result, a_fl, true);
		meta_copy_d2h(&r_val, result, sizeof(G_TYPE), true);
		//mpi reduce
		MPI_Reduce(&r_val, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
#error Unsupported G_TYPE, must be double or float
#endif
	}
	gettimeofday(&end, NULL);
	double time = (end.tv_sec - start.tv_sec) * 1000000.0
			+ (end.tv_usec - start.tv_usec);
	printf("Local partial sum on rank[%d]: [%f]\n", rank, r_val);
	if (rank == 0 && global_sum != check)
		fprintf(stderr,
				"Error, computed dot-product invalid!\n\tExpect: [%f]\tReturn: [%f]\n",
				check, r_val);
	printf(
			"[%d] Tests completed on rank[%d] with matrices of size[%d][%d][%d]\n\t[%f] Total time (us)\n\t[%f] Average time/iteration (us)\n\t[%f] Average time/element (us)\n",
			niters, rank, ni, nj, nk, time, time / niters,
			time / niters / (ni * nj * nk));
	deallocate();

	metaTimersFinish();
	cleanup();

	meta_mpi_finalize();
	return flag;
}
