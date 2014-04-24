#!/bin/bash
rm validate.out
#X=128
#Y=128
#Z=128
#TX=8
#TY=8
#TZ=4
FAIL=0
for X in 3 7 13 29 47 79
do
	for Y in 16 32 64 128 256
	do
		for Z in 12 24 48 96 192
		do
			for TX in 1 2 4 8
			do
				for TY in 1 2 4 8
				do
					for TZ in 1 2 4
					do
						AFOSR_TIMER_LEVEL=1 AFOSR_MODE=CUDA ./red_test $X $Y $Z 1 $TX $TY $TZ &> tempout

						LINES=(`grep "approximate" tempout | sed "s/\[/ /g" | sed "s/\]/ /g" | awk '{print $10}'`)
						((EXPECT=(X-2)*(Y-2)*(Z-2)))
						echo "C_CUDA, $X, $Y, $Z, 1, $TX, $TY, $TZ, $EXPECT, ${LINES[0]}, ${LINES[1]}, ${LINES[2]}, ${LINES[3]}, " >> validate.out
						for TEST in `grep "Test" tempout | sed "s/\./ /g" | awk '{print $3}'`
						do
							if [ $TEST != $EXPECT ]
							then 
								FAIL=1
							fi
						done
						if [ $FAIL == 1 ]
						then echo "error, " >> validate.out
							cat tempout >> validate.out
							FAIL=0
						fi
						AFOSR_TIMER_LEVEL=1 AFOSR_MODE=OpenCL TARGET_DEVICE="Tesla K20c" ./red_test $X $Y $Z 1 $TX $TY $TZ &> tempout

						LINES=(`grep "approximate" tempout | sed "s/\[/ /g" | sed "s/\]/ /g" | awk '{print $10}'`)
						((EXPECT=(X-2)*(Y-2)*(Z-2)))
						echo "C_OCL_GPU, $X, $Y, $Z, 1, $TX, $TY, $TZ, $EXPECT, ${LINES[0]}, ${LINES[1]}, ${LINES[2]}, ${LINES[3]}, " >> validate.out
						for TEST in `grep "Test" tempout | sed "s/\./ /g" | awk '{print $3}'`
						do
							if [ $TEST != $EXPECT ]
							then 
								FAIL=1
							fi
						done
						if [ $FAIL == 1 ]
						then echo "error, " >> validate.out
							cat tempout >> validate.out
							FAIL=0
						fi
						AFOSR_TIMER_LEVEL=1 AFOSR_MODE=OpenCL TARGET_DEVICE="Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz" ./red_test $X $Y $Z 1 $TX $TY $TZ &> tempout

						LINES=(`grep "approximate" tempout | sed "s/\[/ /g" | sed "s/\]/ /g" | awk '{print $10}'`)
						((EXPECT=(X-2)*(Y-2)*(Z-2)))
						echo "C_OCL_CPU, $X, $Y, $Z, 1, $TX, $TY, $TZ, $EXPECT, ${LINES[0]}, ${LINES[1]}, ${LINES[2]}, ${LINES[3]}, " >> validate.out
						for TEST in `grep "Test" tempout | sed "s/\./ /g" | awk '{print $3}'`
						do
							if [ $TEST != $EXPECT ]
							then 
								FAIL=1
							fi
						done
						if [ $FAIL == 1 ]
						then echo "error, " >> validate.out
							cat tempout >> validate.out
							FAIL=0
						fi
						AFOSR_TIMER_LEVEL=1 AFOSR_MODE=CUDA ./red_test_fortran $X $Y $Z 1 $TX $TY $TZ &> tempout

						LINES=(`grep "approximate" tempout | sed "s/\[/ /g" | sed "s/\]/ /g" | awk '{print $10}'`)
						((EXPECT=(X-2)*(Y-2)*(Z-2)))
						echo "F_CUDA, $X, $Y, $Z, 1, $TX, $TY, $TZ, $EXPECT, ${LINES[0]}, ${LINES[1]}, ${LINES[2]}, ${LINES[3]}, " >> validate.out
						for TEST in `grep "Test" tempout | sed "s/\./ /g" | awk '{print $3}'`
						do
							if [ $TEST != $EXPECT ]
							then 
								FAIL=1
							fi
						done
						if [ $FAIL == 1 ]
						then echo "error, " >> validate.out
							cat tempout >> validate.out
							FAIL=0
						fi
						AFOSR_TIMER_LEVEL=1 AFOSR_MODE=OpenCL TARGET_DEVICE="Tesla K20c" ./red_test_fortran $X $Y $Z 1 $TX $TY $TZ &> tempout

						LINES=(`grep "approximate" tempout | sed "s/\[/ /g" | sed "s/\]/ /g" | awk '{print $10}'`)
						((EXPECT=(X-2)*(Y-2)*(Z-2)))
						echo "F_OCL_GPU, $X, $Y, $Z, 1, $TX, $TY, $TZ, $EXPECT, ${LINES[0]}, ${LINES[1]}, ${LINES[2]}, ${LINES[3]}, " >> validate.out
						for TEST in `grep "Test" tempout | sed "s/\./ /g" | awk '{print $3}'`
						do
							if [ $TEST != $EXPECT ]
							then 
								FAIL=1
							fi
						done
						if [ $FAIL == 1 ]
						then echo "error, " >> validate.out
							cat tempout >> validate.out
							FAIL=0
						fi
						AFOSR_TIMER_LEVEL=1 AFOSR_MODE=OpenCL TARGET_DEVICE="Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz" ./red_test_fortran $X $Y $Z 1 $TX $TY $TZ &> tempout

						LINES=(`grep "approximate" tempout | sed "s/\[/ /g" | sed "s/\]/ /g" | awk '{print $10}'`)
						((EXPECT=(X-2)*(Y-2)*(Z-2)))
						echo "F_OCL_CPU, $X, $Y, $Z, 1, $TX, $TY, $TZ, $EXPECT, ${LINES[0]}, ${LINES[1]}, ${LINES[2]}, ${LINES[3]}, " >> validate.out
						for TEST in `grep "Test" tempout | sed "s/\./ /g" | awk '{print $3}'`
						do
							if [ $TEST != $EXPECT ]
							then 
								FAIL=1
							fi
						done
						if [ $FAIL == 1 ]
						then echo "error, " >> validate.out
							cat tempout >> validate.out
							FAIL=0
						fi
					done
				done
			done
		done
	done
done
