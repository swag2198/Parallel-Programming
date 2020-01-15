#include <iostream>
#include <stdio.h>
#include <omp.h>

using namespace std;

int main()
{
	cout<<"No. of processors = "<<omp_get_num_procs()<<endl;
	cout<<"No. of threads = "<<omp_get_max_threads()<<endl;
	cout<<"Parent Process id = "<<omp_get_thread_num()<<endl;

	double wtime = omp_get_wtime();
	// #pragma omp parallel
	// {
	// 	cout<<"Hello OpenMP!"<<endl; //endl may not be atomic
	// }

	int id;

	#pragma omp parallel\
	private(id)
	{
		id = omp_get_thread_num();
		cout<<"Hello OpenMP from Thread "<<id<<endl;
		// printf("Hello OpenMP from Thread %d \n", id); //prints nicely as expected
	}
	wtime = omp_get_wtime() - wtime;
	cout<<"Elapsed time = "<<wtime<<endl;
	return 0;
}