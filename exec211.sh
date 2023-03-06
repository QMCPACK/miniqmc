#!/bin/bash
rm -rf scalabilite_bench211.dat
TIME_FORMAT=%R
max_core=$1

printf 'running on 1 process\n\n'
echo -n "1 1; " >> scalabilite_bench211.dat 
OMP_NUM_THREADS=1 ~/time -p ./build/bin/miniqmc -g "2 1 1" |& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
echo ";" >> scalabilite_bench211.dat
for((i=2; i<=max_core; i=i*2))
do
	printf 'running on %d mpi process and 1 omp threads\n' "$i"
	echo -n "$i 1; " >> scalabilite_bench211.dat
	OMP_NUM_THREADS=1 ~/time -p mpirun -n $i ./build/bin/miniqmc -g "2 1 1" |& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
	echo ";" >> scalabilite_bench211.dat

	printf 'running on 1 mpi process and %d omp threads\n' "$i"
	echo -n "1 $i; " >> scalabilite_bench211.dat
	OMP_NUM_THREADS=$i ~/time -p ./build/bin/miniqmc -g "2 1 1" |& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
	echo ";" >> scalabilite_bench211.dat

	for((k=2; k<i; k=k*2))
	do
		div=$(($i/$k))
		printf 'running on %d mpi process and %d omp threads\n' "$k" "$div"
		echo -n "$k $div; " >> scalabilite_bench211.dat
		OMP_NUM_THREADS=$div ~/time -p mpirun -n $k ./build/bin/miniqmc -g "2 1 1"|& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
		echo ";" >> scalabilite_bench211.dat

	done
	printf "\n"
	if((i==max_core))
	then
		exit 0
	fi
done

printf 'running on %d mpi process and 1 omp threads\n' "$max_core"
echo -n "$max_core 1; " >> scalabilite_bench211.dat
OMP_NUM_THREADS=1 ~/time -p mpirun -n $max_core ./build/bin/miniqmc -g "2 1 1"|& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
echo ";" >> scalabilite_bench211.dat

printf 'running on 1 mpi process and %d omp threads\n' "$max_core"
echo -n "1 $max_core; " >> scalabilite_bench211.dat
OMP_NUM_THREADS=$max_core ~/time -p ./build/bin/miniqmc -g "2 1 1"|& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
echo ";" >> scalabilite_bench211.dat

for((k=2; k<=max_core; k=k*2))
do
	div=$(($max_core/$k))
	if((div == 1 && k!= max_core)); then break; fi
	printf 'running on %d mpi process and %d omp threads\n' "$k" "$div"
	echo -n "$k $div; " >> scalabilite_bench211.dat
	OMP_NUM_THREADS=$div ~/time -p mpirun -n $k ./build/bin/miniqmc -g "2 1 1"|& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
	echo ";" >> scalabilite_bench211.dat

	printf 'running on %d mpi process and %d omp threads\n' "$div" "$k"
	echo -n "$div $k; " >> scalabilite_bench211.dat
	OMP_NUM_THREADS=$k ~/time -p mpirun -n $div ./build/bin/miniqmc -g "2 1 1"|& grep -E "real [0-9][0-9]*\.[0-9][0-9]*" | sed "s/real //g" >> scalabilite_bench211.dat
	echo ";" >> scalabilite_bench211.dat
done
printf "\n"
