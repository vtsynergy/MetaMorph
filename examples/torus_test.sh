#!/bin/bash

#Want to test with and without: MPI, ASYNC, AUTOCONFIG, BACKENDS
#Needs to be run on a node that can compile all backends and plugins
MPI_PROCS="1 2 4"
OUT_DIR=./results/`date +%s`
mkdir -p $OUT_DIR
#OPENMP loop
for openmp in "TRUE" "FALSE"
do
	#CUDA loop
	for cuda in "TRUE" "FALSE"
	do
		#OpenCL loop
		for opencl in "TRUE" "FALSE"
		do
			#MPI loop
			for mpi in "TRUE" "FALSE"
			do
			make clean || true
			USE_TIMERS=TRUE USE_MPI=$mpi USE_FORTRAN=TRUE USE_CUDA=$cuda USE_OPENCL=$opencl USE_OPENMP=$openmp make all examples &> $OUT_DIR/build.openmp"$openmp".cuda"$cuda".opencl"$opencl".mpi"$mpi".log
			RUN_MODES=
			if [ "$cuda" == "TRUE" ]; then RUN_MODES="$RUN_MODES CUDA"; fi
			if [ "$openmp" == "TRUE" ]; then RUN_MODES="$RUN_MODES OpenMP"; fi
			if [ "$opencl" == "TRUE" ]; then RUN_MODES="$RUN_MODES OpenCL"; fi
				#ASYNC loop
				for async in 1 0
				do
					#AUTOCONF loop
					for autoconf in 1 0
					do
						#Sizes loopi
						for size in 1 2
						do
							for mode in $RUN_MODES
							do
								if [ "$mpi" == "TRUE" ]; then
									for mpi_procs in $MPI_PROCS
									do
										METAMORPH_TIMER_LEVEL=1 METAMORPH_MODE=$mode mpirun -n $mpi_procs ./examples/torus_reduce_test $size $size $size 1 $async $autoconf &> $OUT_DIR/run.openmp"$openmp".cuda"$cuda".opencl"$opencl".mpi"$mpi".async"$async".autoconf"$autoconf".size"$size".mode"$mode".nproc"$mpi_procs".log
									done
								else
									METAMORPH_TIMER_LEVEL=1 METAMORPH_MODE=$mode ./examples/torus_reduce_test $size $size $size 1 $async $autoconf &> $OUT_DIR/run.openmp"$openmp".cuda"$cuda".opencl"$opencl".mpi"$mpi".async"$async".autoconf"$autoconf".size"$size".mode"$mode".nproc1.log
								fi
							done
						done #Sizes loop
					done #AUTOCONF loop
				done #ASYNC loop
			done #MPI loop
		done #OpenCL loop
	done #CUDA loop
done #OPENMP loop
