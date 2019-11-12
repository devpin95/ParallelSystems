// LU decomposition - OpenMP
// https://computing.llnl.gov/tutorials/openMP/
//-----------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& numThreads,int& isPrint)
{
	bool isOK = true;

	if(argc < 3) 
	{
		cout << "Arguments:<X> <Y> [<Z>]" << endl;
		cout << "X : Matrix size [X x X]" << endl;
		cout << "Y : Number of threads" << endl;
		cout << "Z = 1: print the input/output matrix if X < 10" << endl;
		cout << "Z <> 1 or missing: does not print the input/output matrix" << endl;
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <=0) 
		{
			cout << "Matrix size must be larger than 0" <<endl;
			isOK = false;
		}

		//get number of threads
		numThreads = atoi(argv[2]);
		if (numThreads <= 0)
		{	cout << "Number of threads must be larger than 0" <<endl;
			isOK = false;
		}

		//is print the input/output matrix
		if (argc >=4)
			isPrint = (atoi(argv[3])==1 && n <=9)?1:0;
		else
			isPrint = 0;
	}
	return isOK;
}

//-----------------------------------------------------------------------
//Initialize the value of matrix a[n x n]
//-----------------------------------------------------------------------
void InitializeMatrix(float** &a,int n)
{
	a = new float*[n]; 
	a[0] = new float[n*n];
	for (int i = 1; i < n; i++)	a[i] = a[i-1] + n;

	#pragma omp parallel for schedule(static) 
	for (int i = 0 ; i < n ; i++)
	{
		for (int j = 0 ; j < n ; j++)
		{	
            if (i == j) 
              a[i][j] = (((float)i+1)*((float)i+1))/(float)2;	
            else
              a[i][j] = (((float)i+1)+((float)j+1))/(float)2;
		}
	}
}

//-----------------------------------------------------------------------
//Initialize a matrix a[n x n] of zeros
//-----------------------------------------------------------------------
void InitializeDiagonalMatrix(float** &a,int n)
{
    a = new float*[n];
    a[0] = new float[n*n];
    for (int i = 1; i < n; i++)	a[i] = a[i-1] + n;

#pragma omp parallel for schedule(static)
    for (int i = 0 ; i < n ; i++)
    {
        for (int j = 0 ; j < n ; j++)
        {
            if ( i == j ) a[i][j] = 1;
            else a[i][j] = 0;
        }
    }
}

//------------------------------------------------------------------
//Delete matrix matrix a[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **a,int n)
{
	delete[] a[0];
	delete[] a; 
}

//------------------------------------------------------------------
//Print matrix	
//------------------------------------------------------------------
void PrintMatrix(float **a, int n) 
{
	for (int i = 0 ; i < n ; i++)
	{
		cout<< "Row " << (i+1) << ":\t" ;
		for (int j = 0 ; j < n ; j++)
		{
			printf("%.2f\t", a[i][j]);
		}
		cout<<endl ;
	}
}

//------------------------------------------------------------------
//Compute the Gaussian Elimination for matrix a[n x n]
//------------------------------------------------------------------
bool ComputeLUDecomposition(float **L, float **U, int n)
{
    // do row wise gaussian elimination on U

    bool is_singular = false;   // flag if we encounter a column of all 0's

    int pivot;  // the value to multiply the row by
    int i;      // the rows we are looking at
    int j;      // the columns that need to be swapped if we pivot
    int k;      // the column we are looking at

    float globalmax;        // the global max for pivoting
    int globalmaxindex;     // the row index of the global max
    float privatemax;       // the max that the thread has encountered
    int privateindex;       // the index of the thread max (used for max and pivoting)
    float privatetemp;      // the value in the column that needs to be checked as the max

    int row_pointer;     // The pointer to the next row that needs to be checked as the max
    int column_pointer;
    int tid;

    for ( k = 0; k < n; ++k ) {
        #pragma omp parallel shared(L, U, globalmax, globalmaxindex, row_pointer, column_pointer) firstprivate(n, k) private(i, j, pivot, privatemax, privateindex, tid)
        {
            tid = omp_get_thread_num();
            #pragma omp single
            {
                // Set the row pointer to the first column that needs to be looked at
                // this will always be the same as the column variable k.
                // set the column pointer to k if we need to pivot. Set it to k because all columns before k will
                // be 0 and we dont need to swap them.
                // reset the global max to 0.0 as well.
                row_pointer = k;
                column_pointer = k;
                globalmax = 0.0;
            }

            // make sure the thread's private max value is reset to 0.0
            privatemax = 0.0;

            // find the pivot point
            while ( row_pointer < n ) {
                #pragma omp critical
                {
                    privateindex = row_pointer;
                    ++row_pointer;
                }

                // check if all of the rows have already been checked
                // if they have then the thread needs to skip checking the index
                if ( privateindex >= n ) break;

                // otherwise, we need to get the absolute value of the elements at U[privateindex][k]
                privatemax = abs(U[privateindex][k]);

                // now we need to compare it to the global value
                #pragma omp critical
                {
                    if ( globalmax < privatemax ) {
                        globalmax = privatemax;
                        globalmaxindex = privateindex;
                    }
                }

            // check the row pointer at the end so that the thread doesnt go back through
            // we dont need to worry about a critical section because we also check it at the beginning
            }

            // make sure none of the threads get past this point until all of the elements in the column have
            // been checked if they are the max
            #pragma omp barrier

            // have the master check if the max is 0.0
            // if it is 0.0, then the column is all 0's and the matrix is singular
            // the rest of the threads can continue ahead to pivoting
            #pragma omp master
            {
                // if the global max is still 0.0, then the column is all 0's
                if ( globalmax == 0.0 ) {
                    is_singular = true;
                }
            }

            if ( !is_singular ) {
                // check if the globalmax was at the row that we are trying to pivot
                if (globalmaxindex != k) {
                    // the index is not the row we are on so we need to swap it to the current row
                    while (column_pointer < n) {
                        #pragma omp critical
                        {
                            privateindex = column_pointer;
                            ++column_pointer;
                        }

                        // check if all of the rows have already been checked
                        // if they have then the thread needs to skip checking the index
                        if (privateindex >= n) break;

                        // swap the current row with the row with the max value
                        float top = U[k][privateindex];
                        float bottom = U[globalmaxindex][privateindex];
                        U[k][privateindex] = bottom;
                        U[globalmaxindex][privateindex] = top;


                        // check the column pointer at the end so that the thread doesnt go back through
                        // we dont need to worry about a critical section because we also check it at the beginning
                    }
                }
            }

//            #pragma omp single
//            {
//                cout << "Column " << k << " max " << globalmax << " in row " << globalmaxindex << endl;
//            }
        }
    }

    return !is_singular;
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	int n,numThreads,isPrintMatrix;
	float **a, **L, **U;
	double runtime;
	bool isOK;
	
	if (GetUserInput(argc,argv,n,numThreads,isPrintMatrix)==false) return 1;

	//specify number of threads created in parallel region
	omp_set_num_threads(numThreads);

	//Initialize the value of matrix A[n x n]
	InitializeMatrix(a,n);
    InitializeMatrix(U, n);
    InitializeDiagonalMatrix(L, n);


	if (isPrintMatrix)
	{
//		cout<< "The input matrix" << endl;
//		PrintMatrix(a,n);
//
//        cout<< "The L matrix" << endl;
//        PrintMatrix(L,n);

        cout<< "The U matrix" << endl;
        PrintMatrix(U,n);
	}

	runtime = omp_get_wtime();

	//Compute the Gaussian Elimination for matrix a[n x n]
	isOK = ComputeLUDecomposition(L, U, n);

	runtime = omp_get_wtime() - runtime;

	if (isOK == true)
	{
		//The eliminated matrix is as below:
		if (isPrintMatrix)
		{
			cout<< "Output matrix:" << endl;
			PrintMatrix(U,n);
		}

		//print computing time
		cout<< "Gaussian Elimination runs in "	<< setiosflags(ios::fixed)
												<< setprecision(2)
												<< runtime << " seconds\n";
	}
	else
	{
		cout<< "The matrix is singular" << endl;
	}

    // the code will run according to the number of threads specified in the arguments
    cout << "Matrix LU decomposition is computed using max of threads = "<< omp_get_max_threads() << " threads or cores" << endl;

    cout << "Matrix size  = " << n << endl;
    
	DeleteMatrix(a,n);
	DeleteMatrix(L, n);
	DeleteMatrix(U, n);
	return 0;
}
