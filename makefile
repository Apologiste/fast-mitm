all: mitmc_parallel

mitmc_parallel: mitmc_parallel.c
	mpicc -o mitmc_parallel mitmc_parallel.c -lm -fopenmp -O3 -march=native

clean:
	rm -f mitmc_parallel
