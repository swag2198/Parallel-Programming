/* 
Name 1: Soumava Paul, Roll:16EE10056
Name 2: Swagatam Haldar, Roll:16EE10063
Compile: gcc -o executable_name 16EE10056_16EE10063.c -lm
Run: ./executable_name number_of_threads filename4arbitraryAxis filename4objectFile angle_OF_rotation_in_degress
We have implemented the rotation functions using parallelized matrix multiplications.
The matrices are sequentially multiplied with the object matrix (3, num_pts) as required by the transformation.
Both Serial and Parallel Implementations are given and their times compared.
Output matrix saved to output.txt
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

#include <omp.h>

#define PI acos(-1.0)


double** give_object(char filename[], int* col) //shape = 3 x N
{
	int countobj = 0, i, j;
	double x;
	FILE* obj;
	obj = fopen(filename, "r");
	while(fscanf(obj, "%lf", &x) == 1)
		countobj++;
	countobj /= 3;
	*col = countobj;
	// printf("countcol = %d\n", *col);
	fclose(obj);

	double **object = (double **)malloc(3 * sizeof(double *));
	for(i=0; i<3; i++)
		object[i] = (double *)malloc(countobj * sizeof(double));

	i = 0;
	j = 0;
	obj = fopen(filename, "r");
	while(fscanf(obj, "%lf", &x) == 1)
	{
		// printf("%d %d started\n",i, j);
		// printf("x = %lf\n",x);
		object[i++][j] = x;
		// printf("%d %d done\n",i, j);
		if(i != 0 && i%3 == 0)
		{
			j++;
			i = 0;
		}
	}
	fclose(obj);
	// printf("gfg");
	return object;
}

double** give_axis(char axis[])
{
	double x;
	int i;
	char PQ[50];
	FILE *ax = fopen(axis, "r");
	fgets(PQ, 50, ax);
    fclose(ax);
	char *p = PQ;
	double **line = (double **)malloc(2*sizeof(double *));
	for(i=0; i<2; i++)
		line[i] = (double *)malloc(3*sizeof(double));
	i=0; 
	int j=0;
	while(*p){
		// printf("pq = %s\n", PQ);
		if (isdigit(*p) || ((*p=='-'||*p=='.') && isdigit(*(p+1)))){
	        // Found a number
	        x = strtod(p, &p);
	        line[i][j++] = x;
	        if(j != 0 && j%3 == 0){
				i++;
				j = 0;
			}
	    } 
    	else p++;
	}	
	/*
	return a 2x3 double matrix containign P and Q as points.
	*/
	return line;
}

void print_object(double** obj, int col)
{
	for(int j=0; j<col; j++)
		printf("j = %d: x = %lf y = %lf z = %lf\n", j+1, obj[0][j], obj[1][j], obj[2][j]);
	printf("\n");
}

//Theta is same for all points. But alpha and beta will be different for each object points.

double** translate_allp(double** object, int n, double P[])
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));


	#pragma omp parallel shared(result, object)
	{
		#pragma omp for schedule(dynamic, 100)
			for(int j=0; j<n; j++)
				for(int i=0; i<3; i++)
					result[i][j] = object[i][j] - P[i];
	}

	return result;
}

double** translate_all(double** object, int n, double P[])
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));

	for(int j=0; j<n; j++)
		for(int i=0; i<3; i++)
			result[i][j] = object[i][j] - P[i];


	return result;
}

double** Rx(double** object, int n, double deg) //takes 3 x N object
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));
	double theta = (PI / 180.0) * deg;
	// printf("thetax = %lf\n", deg);

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			result[i][j] = 0;

	double rotx[3][3] = {
		{1, 0, 0},
		{0, cos(theta), -sin(theta)},
		{0, sin(theta), cos(theta)}
	};

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			for(int k=0; k<3; k++)
				result[i][j] += rotx[i][k] * object[k][j];
	return result;
}

double** Rxp(double** object, int n, double deg) //takes 3 x N object
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));
	double theta = (PI / 180.0) * deg;
	// printf("thetax = %lf\n", deg);

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			result[i][j] = 0;

	double rotx[3][3] = {
		{1, 0, 0},
		{0, cos(theta), -sin(theta)},
		{0, sin(theta), cos(theta)}
	};

	#pragma omp parallel shared(result, rotx, object) 
	{
		// printf("num_threads = %d from Rx\n", omp_get_num_threads());
		// #pragma omp for schedule (dynamic, 10)
		#pragma omp for collapse(2) schedule (dynamic, 100)
			for(int i=0; i<3; i++)
				for(int j=0; j<n; j++)
					for(int k=0; k<3; k++)
						result[i][j] += rotx[i][k] * object[k][j];
	}
	return result;
}

double** Ry(double** object, int n, double deg) //takes 3 x N object
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));
	double theta = (PI / 180.0) * deg;
	// printf("thetay = %lf\n", deg);

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			result[i][j] = 0;

	double roty[3][3] = {
		{cos(theta), 0, sin(theta)},
		{0, 1, 0},
		{-sin(theta), 0, cos(theta)}
	};

	int r = 3, c = n;

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			for(int k=0; k<3; k++)
				result[i][j] += roty[i][k] * object[k][j];
	return result;
}

double** Ryp(double** object, int n, double deg) //takes 3 x N object
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));
	double theta = (PI / 180.0) * deg;
	// printf("thetay = %lf\n", deg);

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			result[i][j] = 0;

	double roty[3][3] = {
		{cos(theta), 0, sin(theta)},
		{0, 1, 0},
		{-sin(theta), 0, cos(theta)}
	};

	int r = 3, c = n;

	#pragma omp parallel shared(result, roty, object)
	{
		// printf("num_threads = %d from Ry\n", omp_get_num_threads());
		// #pragma omp for schedule (dynamic, 10)
		#pragma omp for collapse(2) schedule (dynamic, 100)
			for(int i=0; i<3; i++)
				for(int j=0; j<n; j++)
					for(int k=0; k<3; k++)
						result[i][j] += roty[i][k] * object[k][j];
	}
	
	return result;
}

double** Rz(double** object, int n, double deg) //takes 3 x N object
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));
	double theta = (PI / 180.0) * deg;
	// printf("thetaz = %lf\n", deg);

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			result[i][j] = 0;

	double rotz[3][3] = {
		{cos(theta), -sin(theta), 0},
		{sin(theta), cos(theta), 0},
		{0, 0, 1}
	};

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			for(int k=0; k<3; k++)
				result[i][j] += rotz[i][k] * object[k][j];
	return result;
}

double** Rzp(double** object, int n, double deg) //takes 3 x N object
{
	double** result = (double **)malloc(3 * sizeof(double *));
	for(int i=0; i<3; i++)
		result[i] = (double *)malloc(n * sizeof(double));
	double theta = (PI / 180.0) * deg;
	// printf("thetaz = %lf\n", deg);

	for(int i=0; i<3; i++)
		for(int j=0; j<n; j++)
			result[i][j] = 0;

	double rotz[3][3] = {
		{cos(theta), -sin(theta), 0},
		{sin(theta), cos(theta), 0},
		{0, 0, 1}
	};

	#pragma omp parallel shared(result, rotz, object)
	{
		// printf("num_threads = %d from Rz\n", omp_get_num_threads());
		// #pragma omp for schedule (dynamic, 200)
		#pragma omp for collapse(2) schedule (dynamic, 100)
			for(int i=0; i<3; i++)
				for(int j=0; j<n; j++)
					for(int k=0; k<3; k++)
						result[i][j] += rotz[i][k] * object[k][j];
	}
		

	return result;
}


int main(int argc, char* argv[])
{
	if(argc != 5)
	{
		printf("all arguments not supplied!\n");
		return 0;
	}
	char *_;
	int num_threads = atoi(argv[1]);
	double theta = strtod(argv[4], &_);

	printf("num_threads = %d\n", num_threads);
	printf("obj filename = %s\n", argv[3]);
	printf("axis filename = %s\n", argv[2]);
	printf("theta = %.3f\n", theta);

	omp_set_num_threads(num_threads);

	int cols;
	double** object = give_object(argv[3], &cols);
	double** axis = give_axis(argv[2]);

	printf("Number of object points = %d\n", cols);

	double* P = axis[0];
	double* Q = axis[1];
	double minusP[3] = {-P[0], -P[1], -P[2]};

	double **r1, **r2, **r3, **r4, **r5, **r6, **r7, **r7_p;
	double x, y, z;
	x = Q[0] - P[0];
	y = Q[1] - P[1];
	z = Q[2] - P[2];

	double alpha = atan(x/z);
	alpha = (180.0 / PI) * alpha;
	printf("alpha = %lf\n", alpha);
	double beta = atan(y / sqrt(x*x + z*z));
	beta = (180.0 / PI) * beta;
	printf("beta = %lf\n", beta);

	double t1, t2;

	t1 = omp_get_wtime();

	r1 = translate_all(object, cols, P);
	// printf("r1 = \n");
	// print_object(r1, cols);
	r2 = Ry(r1, cols, -alpha);
	// printf("r2 = \n");
	// print_object(r2, cols);
	r3 = Rx(r2, cols, beta);
	// printf("r3 = \n");
	// print_object(r3, cols);
	r4 = Rz(r3, cols, theta);
	// printf("r4 = \n");
	// print_object(r4, cols);
	r5 = Rx(r4, cols, -beta);
	// printf("r5 = \n");
	// print_object(r5, cols);
	r6 = Ry(r5, cols, alpha);
	// printf("r6 = \n");
	// print_object(r6, cols);
	r7 = translate_all(r6, cols, minusP);

	t2 = omp_get_wtime();

	// printf("r7 = \n");
	// print_object(r7, cols);
	printf("Total serial time taken = %lf seconds\n", t2-t1);

	t1 = omp_get_wtime();

	r1 = translate_allp(object, cols, P);
	// printf("r1 = \n");
	// print_object(r1, cols);
	r2 = Ryp(r1, cols, -alpha);
	// printf("r2 = \n");
	// print_object(r2, cols);
	r3 = Rxp(r2, cols, beta);
	// printf("r3 = \n");
	// print_object(r3, cols);
	r4 = Rzp(r3, cols, theta);
	// printf("r4 = \n");
	// print_object(r4, cols);
	r5 = Rxp(r4, cols, -beta);
	// printf("r5 = \n");
	// print_object(r5, cols);
	r6 = Ryp(r5, cols, alpha);
	// printf("r6 = \n");
	// print_object(r6, cols);
	r7_p = translate_allp(r6, cols, minusP);

	t2 = omp_get_wtime();
	printf("Total parallel time taken = %lf seconds\n", t2-t1);
	// printf("r7 = \n");
	// print_object(r7_p, cols);
	int flag=0;
	for(int i=0; i<3;++i){
		for(int j=0;j<cols;++j){
			if(abs(r7_p[i][j]-r7[i][j])>1e-5) flag=1;
		}
	}
	
	if(flag) printf("Parallelized Output does not match Serial\n");
	else printf("Parallelized Output matches Serial\n");

	FILE* out_file = fopen("output.txt", "w");
	for(int j=0; j<cols; ++j){
		for(int i=0; i<3; ++i){
			fprintf(out_file, "%lf\t", r7[i][j]);
		}
		fprintf(out_file, "\n");
	}

	printf("output matrix written to output.txt\n");
	
	fclose(out_file);

	return 0;
}