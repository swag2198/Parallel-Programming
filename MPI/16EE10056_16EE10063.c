#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "png.h"

int* give_histogram(int* img, int n)
{
	int *hist = (int *)malloc(256 * sizeof(int));
	for(int i=0; i<256; i++)
		hist[i] = 0;
	for(int i=0; i<n; i++)
		hist[img[i]] += 1;
	return hist;
}

int isvalid(int i, int j, int w, int num)
{
	int index = i*w+ j;
	if(index < 0 || index >= num)
		return 0;
	return 1;
}

int main()
{
	MPI_Init(NULL, NULL);

	int i,j, temp=0, myid, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	FILE *fp = fopen("sample2.png", "r");
	int *matrix;
	
	png_structp pngptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop pnginfo = png_create_info_struct(pngptr);
	png_init_io(pngptr, fp);
	png_bytepp rows;
	png_read_png(pngptr, pnginfo, PNG_TRANSFORM_IDENTITY, NULL);
	
	rows = png_get_rows(pngptr, pnginfo);
	int width = png_get_image_width(pngptr, pnginfo);
	int height = png_get_image_height(pngptr, pnginfo);
	//printf("\nSize of Image = %d x %d\n", width, height);

	if(myid == 0)
	{
		//Process 0 creates the Image array into matrix
		matrix = (int *)malloc(height * width * sizeof(int *));
	
		for (i=0;i<height;i++){
			for(j=0;j<width;j++){
				matrix[i*width + j]=(rows[i][j]+rows[i][width+j]+rows[i][2*width+j])/3;
			}
		}
	}
	printf("Image matrix ready\n");
	MPI_Barrier(MPI_COMM_WORLD);

	int num_elements = height * width;
	int num_elements_per_proc = num_elements / size;
	num_elements_per_proc = (height / size) * width;
	int level = 256;

	int *sub_matrix = (int *)malloc(num_elements_per_proc * sizeof(int));
	MPI_Scatter(matrix, num_elements_per_proc, MPI_INT, sub_matrix, num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
	printf("Image sub matrices scattered\n");

	int *sub_hist = give_histogram(sub_matrix, num_elements_per_proc);
	MPI_Barrier(MPI_COMM_WORLD);

	int *sub_hists = (int *)malloc(level * size * sizeof(int));
	MPI_Allgather(sub_hist, level, MPI_INT, sub_hists, level, MPI_INT, MPI_COMM_WORLD);
	printf("all gather complete\n");
	int *S = (int *)malloc(level * sizeof(int)); //Sk gives transformed intensity

	if(myid == 0)
	{
		//Process 0 combines the sub histograms
		int *hist = (int *)malloc(level * sizeof(int));
		int *prefix = (int *)malloc(level * sizeof(int));
		for(i=0; i<level; i++)
		{
			hist[i] = 0;
			prefix[i] = 0;
		}
		for(i=0; i<level; i++)
		{
			for(j=0; j<size; j++)
				hist[i] += sub_hists[j*size + i];
		}
		//Process 0 also computes the intensity transformation Sk
		prefix[0] = hist[0];
		for(i=1; i<level; i++)
			prefix[i] += prefix[i-1] + hist[i];

		double tf;
		for(i=0; i<level; i++)
		{
			tf = 255.0 / (height * width);
			tf *= prefix[i];
			S[i] = floor(tf);
		}
	}
	//P0 broadcasts the transformation to the world
	MPI_Bcast(S, level, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("transformation broadcast\n");
	//in place modification of sub matrices present in each process
	int old, new;
	for(int i=0; i<num_elements_per_proc; i++)
	{
		old = sub_matrix[i];
		new = S[old];
		sub_matrix[i] = new;
	}
	int *newmat;
	if(myid == 0)
	{
		//Initialise new matrix to store equalised image
		newmat = (int *)malloc(num_elements_per_proc * size * sizeof(int *));
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(sub_matrix, num_elements_per_proc, MPI_INT, newmat, num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

	if(myid == 0)
	{
		//Store the final image as PGM file
		FILE* pgmimg; 
	    pgmimg = fopen("sample1.pgm", "wb"); 
	  
	    // Writing Magic Number to the File 
	    fprintf(pgmimg, "P2\n");  
	  
	    // Writing Width and Heigh
	    fprintf(pgmimg, "%d %d\n", width, height);  
	  
	    // Writing the maximum gray value 
	    fprintf(pgmimg, "255\n");  
	    int count = 0; 
	    for (i=0; i<height; i++) { 
	        for (j=0; j<width; j++) { 
	            temp = newmat[i*width+j]; 
	            // Writing the gray values in the 2D array to the file 
	            fprintf(pgmimg, "%d ", temp); 
	        } 
	        fprintf(pgmimg, "\n"); 
	    } 
	    fclose(pgmimg);
	}

	//Sobel Computation on submatrices starts here
	int *sub_sobelx = (int *)malloc(num_elements_per_proc * sizeof(int));
	int *sub_sobely = (int *)malloc(num_elements_per_proc * sizeof(int));
	int *sub_sobel = (int *)malloc(num_elements_per_proc * sizeof(int));

	int rows_per_proc = height / size;
	int a,b,c,d,e,f, x, y;
	for(int i=0; i<rows_per_proc; i++)
	{
		for(int j=0; j<width; j++)
		{
			a = b = c = d = e = f = 0;
			if(isvalid(i+1,j-1,width,num_elements_per_proc))
				a = sub_matrix[(i+1)*width + j-1];
			if(isvalid(i-1,j-1,width,num_elements_per_proc))
				b = sub_matrix[(i-1)*width + j-1];
			if(isvalid(i+1,j,width,num_elements_per_proc))
				c = sub_matrix[(i+1)*width + j];
			if(isvalid(i-1,j,width,num_elements_per_proc))
				d = sub_matrix[(i-1)*width + j];
			if(isvalid(i+1,j+1,width,num_elements_per_proc))
				e = sub_matrix[(i+1)*width + j+1];
			if(isvalid(i-1,j+1,width,num_elements_per_proc))
				f = sub_matrix[(i-1)*width + j+1];

			sub_sobelx[width*i + j] = (a-b) + 2*(c-d)  + (e-f);
			x = sub_sobelx[width*i + j];

			a = b = c = d = e = f = 0;
			if(isvalid(i-1,j+1,width,num_elements_per_proc))
				a = sub_matrix[(i-1)*width + j+1];
			if(isvalid(i-1,j-1,width,num_elements_per_proc))
				b = sub_matrix[(i-1)*width + j-1];
			if(isvalid(i,j+1,width,num_elements_per_proc))
				c = sub_matrix[(i)*width + j+1];
			if(isvalid(i,j-1,width,num_elements_per_proc))
				d = sub_matrix[(i)*width + j-1];
			if(isvalid(i+1,j+1,width,num_elements_per_proc))
				e = sub_matrix[(i+1)*width + j+1];
			if(isvalid(i+1,j-1,width,num_elements_per_proc))
				f = sub_matrix[(i+1)*width + j-1];

			sub_sobely[width*i + j] = (a-b) + 2*(c-d)  + (e-f);
			y = sub_sobely[width*i + j];


			sub_sobel[width*i + j] = sqrt(x*x + y*y);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	printf("sub sobels ready\n");

	int *sobel;
	if(myid == 0)
	{
		//Initialise sobel to combine sub sobels
		sobel = (int *)malloc(num_elements_per_proc * size * sizeof(int *));
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(sub_sobel, num_elements_per_proc, MPI_INT, sobel, num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

	if(myid == 0)
	{
		//Store the final sobel image as PGM file
		FILE* pgm1; 
	    pgm1 = fopen("sample1_sobel.pgm", "wb"); 
	  
	    // Writing Magic Number to the File 
	    fprintf(pgm1, "P2\n");  
	  
	    // Writing Width and Heigh
	    fprintf(pgm1, "%d %d\n", width, height);  
	  
	    // Writing the maximum gray value 
	    fprintf(pgm1, "255\n");  
	    int count = 0; 
	    for (i=0; i<height; i++) { 
	        for (j=0; j<width; j++) { 
	            temp = sobel[i*width+j]; 
	            // Writing the gray values in the 2D array to the file 
	            fprintf(pgm1, "%d ", temp); 
	        } 
	        fprintf(pgm1, "\n"); 
	    } 
	    fclose(pgm1);
	}

	MPI_Finalize();
	return 0;
}



	
	