#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define MAXIMUM_VALUE   1000000.0f
#define HANDLE_ERROR( err )  ( HandleError( err, __FILE__, __LINE__ ) )
#define MAX_BLOCK_DIM 1024
#define MAX_BLOCKS 56

typedef struct {
    double min;
    double max;
    double sum;
} local_mms;

void HandleError( cudaError_t err, const char *file, int line ) {
  //
  // Handle and report on CUDA errors.
  //
  if ( err != cudaSuccess ) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );

    exit( EXIT_FAILURE );
  }
}

void checkCUDAError( const char *msg, bool exitOnError ) {
  //
  // Check cuda error and print result if appropriate.
  //
  cudaError_t err = cudaGetLastError();

  if( cudaSuccess != err) {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      if (exitOnError) {
        exit(-37);
      }
  }                         
}

void cleanupCuda( void ) {
    //
    // Clean up CUDA resources.
    //

    //
    // Explicitly cleans up all runtime-related resources associated with the
    // calling host thread.
    //
    HANDLE_ERROR(
         cudaThreadExit()
    );
}

__device__ double device_pow( double x, double y ) {
    //
    // Calculate x^y on the GPU.
    //
    return pow( x, y );
}

__device__ double device_ceil( double x ) {
    //
    // Calculate x^y on the GPU.
    //
    return ceilf( x );
}

//
// PLACE GPU KERNELS HERE - BEGIN
//
__global__ void k_sum( double *A, unsigned int *size ) {
    extern __shared__ double sdata[];

    // get the size of the array
    unsigned int N = (*size) ;

    unsigned int tid = threadIdx.x;
    unsigned int num_levels = device_ceil( (double)gridDim.x / blockDim.x );

    if (num_levels == 1) num_levels = 0;

    for ( int level = 0; level <= num_levels; ++level ) {
        // find the index in the global array that the thread needs to load
        int i = device_pow(blockDim.x, level) * ( ( blockIdx.x * blockDim.x ) + tid );

        if ( i < N ) sdata[tid] = A[i]; // if the index is in range, load element from global mem
        else sdata[tid] = 0; // otherwise, just make it 0

        int old_s = blockDim.x; // remember how many elements are in the whole partition

        __syncthreads();

        // partition the array into upper and lower half
        // pair each lower element with an upper element
        // if old_s is odd (the number of elements in the current partition) then there will be an element
        // (the last element in the upper half) wont get a pair in the lower half so we need to add it manually
        for( int s = blockDim.x/2; s > 0; s >>= 1 ) {
            if ( tid < s ) sdata[tid] += sdata[tid + s];  // if in the lower half add to it's pair
            else if ( tid == 2 * s && old_s % 2 == 1 ) sdata[0] += sdata[s * 2]; // the last element didnt get a pair
            old_s /= 2; // now we will focus on the lower half
            __syncthreads();
        }

        if ( tid == 0 ) A[i] = sdata[tid];
        //else A[i] = -1.0;
        __syncthreads();
    }
}

__global__ void k_sum_gross( double *A, unsigned int *size ) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + tid;
    unsigned int N = (*size);

    unsigned int num_levels = device_ceil( (double)gridDim.x / blockDim.x );

    if ( gridDim.x == 1 ) {
        num_levels = 0;
    }

    unsigned int active_blocks = gridDim.x;

    __syncthreads();

    for ( int level = 0; level <= num_levels; ++level ) {

        if ( blockIdx.x < active_blocks ) {
            if (i < N) sdata[tid] = A[i];
            else sdata[tid] = 0;

            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            __syncthreads();


            A[i] = 0.0;
            __syncthreads();
            if (tid == 0) A[blockIdx.x] = sdata[tid];

            __syncthreads();
        }

        N = active_blocks;
        active_blocks = device_ceil((double)active_blocks / blockDim.x);
        __syncthreads();
    }
}

__global__ void k_sum_blocked( double *A, local_mms *B, unsigned int *size ) {
    extern __shared__ double sdata[];

    __shared__ double multiblocksum;
    __shared__ double local_min;
    __shared__ double local_max;

    unsigned int mindata_offset = blockDim.x;
    unsigned int maxdata_offset = blockDim.x * 2;

    unsigned int tid = threadIdx.x;
    unsigned int N = (*size);

    unsigned int active_block = blockIdx.x;
    unsigned int step = 0;
    unsigned int partition_start_index = ( active_block + ( step * MAX_BLOCKS ) ) * blockDim.x;

    if ( tid == 0 ) {
        multiblocksum = 0.0;
        local_min = DBL_MAX;
        local_max = 0;
    }

    while ( partition_start_index < N ) {
        unsigned int current_partition = active_block + ( step * MAX_BLOCKS );
        unsigned int i = blockDim.x * current_partition + tid;

        if (i < N) {
            double Ai = A[i]; // only copy it from global memory once instead of 3 times
            sdata[tid] = Ai;
            sdata[tid + mindata_offset] = Ai;
            sdata[tid + maxdata_offset] = Ai;
        }
        else {
            sdata[tid] = 0;
            sdata[tid + mindata_offset] = -1.0; // -1.0 is a dummy value for an index outside the array
            sdata[tid + maxdata_offset] = -1.0;
        }

        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];

                __syncthreads();
                if ( sdata[tid + mindata_offset] != -1.0 && sdata[tid + s + mindata_offset] != -1.0 ) {
                    if ( sdata[tid + s + mindata_offset] < sdata[tid + mindata_offset] ) {
                        sdata[tid + mindata_offset] = sdata[tid + s + mindata_offset];
                    }
                }
                __syncthreads();
                if ( sdata[tid + maxdata_offset] != -1.0 && sdata[tid + s + maxdata_offset] != -1.0 ) {
                    if ( sdata[tid + s + maxdata_offset] > sdata[tid + maxdata_offset] ) {
                        sdata[tid + maxdata_offset] = sdata[tid + s + maxdata_offset];
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();

        if ( tid == 0) {
            multiblocksum += sdata[0];
            if ( sdata[0 + mindata_offset] < local_min ) {
                local_min = sdata[0 + mindata_offset];
            }
            if ( sdata[0 + maxdata_offset] > local_max ) {
                local_max = sdata[0 + maxdata_offset];
            }
        }

        ++step;
        partition_start_index = ( active_block + ( step * MAX_BLOCKS ) ) * blockDim.x;
    }

    __syncthreads();

    if ( tid == 0 ) {
        B[blockIdx.x].sum = multiblocksum;
        B[blockIdx.x].min = local_min;
        B[blockIdx.x].max = local_max;
    }
}

__global__ void k_sum_stddev( double *A, local_mms *B, unsigned int *size, double *mean ) {
    extern __shared__ double sdata[];

    __shared__ double multiblocksum;
    __shared__ unsigned int step ;


    unsigned int tid = threadIdx.x;
    unsigned int N = (*size);

    unsigned int active_block = blockIdx.x;
    unsigned int partition_start_index = ( active_block + ( step * MAX_BLOCKS ) ) * blockDim.x;

    if ( tid == 0 ) {
        multiblocksum = 0.0;
        step = 0;
    }

    while ( partition_start_index < N ) {
        unsigned int current_partition = active_block + ( step * MAX_BLOCKS );
        unsigned int i = blockDim.x * current_partition + tid;

        if (i < N) sdata[tid] = device_pow( A[i] - (*mean), 2 );
        else sdata[tid] = 0;

        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        __syncthreads();

        if ( tid == 0) {
            multiblocksum += sdata[0];
            ++step;
        }

        partition_start_index = ( active_block + ( step * MAX_BLOCKS ) ) * blockDim.x;
    }

    __syncthreads();

    if ( tid == 0 ) B[blockIdx.x].sum = multiblocksum;
}

//
// PLACE GPU KERNELS HERE - END
//

int main( int argc, char* argv[] ) {
    //
    // Determine min, max, mean, mode and standard deviation of array
    //
    unsigned int array_size, seed, i;
    struct timeval start, end;
    double runtime;

    if( argc < 3 ) {
        printf( "Format: stats_gpu <size of array> <random seed>\n" );
        printf( "Arguments:\n" );
        printf( "  size of array - This is the size of the array to be generated and processed\n" );
        printf( "  random seed   - This integer will be used to seed the random number\n" );
        printf( "                  generator that will generate the contents of the array\n" );
        printf( "                  to be processed\n" );

        exit( 1 );
    }

    //
    // Get the size of the array to process.
    //
    array_size = atoi( argv[1] );

    //
    // Get the seed to be used
    //
    seed = atoi( argv[2] );

    //
    // Make sure that CUDA resources get cleaned up on exit.
    //
    atexit( cleanupCuda );

    //
    // Record the start time.
    //
    gettimeofday( &start, NULL );

    //
    // Allocate the array to be populated.
    //
    //double *array = (double *) malloc( array_size * sizeof( double ) );
    double *array = (double *) malloc( array_size * sizeof(double) );

    //
    // Seed the random number generator and populate the array with its values.
    //
    srand( seed );
    for( i = 0; i < array_size; i++ ) {
        //array[i] = i*2;
        array[i] = ( (double) rand() / (double) RAND_MAX ) * MAXIMUM_VALUE;
        //printf("%4.6f , ", array[i]);
    }
    printf("\n");

    //
    // Setup output variables to hold min, max, mean, and standard deviation
    //
    // YOUR CALCULATIONS BELOW SHOULD POPULATE THESE WITH RESULTS
    //
    double min = DBL_MAX;
    double max = 0;
    double sum = 0;
    double mean = 0;
    double stddev = 0;

    // allocate a new array that has the number of elements we are sending to the gpu
    double *a_sum = (double *) malloc( array_size * sizeof( double ) );
    local_mms *B = (local_mms *) malloc( MAX_BLOCKS * sizeof( local_mms ) );

    size_t total_mem = 0, free_mem = 0;
    HANDLE_ERROR( cudaMemGetInfo(&free_mem, &total_mem) );

    // -----------------------------------------------------------------------------------------------------------------
    // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - BEGIN
    // -----------------------------------------------------------------------------------------------------------------

    double *dA, *dM;
    local_mms *dB;
    unsigned int *device_array_size;

//    HANDLE_ERROR( cudaMalloc( (void **)&device_array_size, sizeof(double) ) );
//    cudaMemcpy(device_array_size, &array_size, sizeof(double), cudaMemcpyHostToDevice);

    //int size = array_size * sizeof(double);


    // MEAN ------------------------------------------------------------------------------------------------------------

    // alloc and cpy array
    HANDLE_ERROR( cudaMalloc( (void **)&dA, array_size * sizeof(double) ) );
    HANDLE_ERROR( cudaMemcpy( dA, array, array_size * sizeof(double), cudaMemcpyHostToDevice ) );

    // alloc and cpy array for gpu output
    HANDLE_ERROR( cudaMalloc( (void **)&dB, MAX_BLOCKS * sizeof(local_mms) ) );
    HANDLE_ERROR( cudaMemcpy( dB, array, MAX_BLOCKS * sizeof(local_mms), cudaMemcpyHostToDevice ) );

    // alloc and cpy array size
    HANDLE_ERROR( cudaMalloc( (void **)&device_array_size, sizeof(unsigned int) ) );
    HANDLE_ERROR( cudaMemcpy( device_array_size, &array_size, sizeof(unsigned int), cudaMemcpyHostToDevice) );


    // KERNEL
    // run the array sum kernel and check for errors
    unsigned int numblocks = (array_size / MAX_BLOCK_DIM) + ((array_size % MAX_BLOCK_DIM ) ? 1 : 0 ) ;
    unsigned int numthreads = MAX_BLOCK_DIM;
    if ( array_size <= numthreads ) {
        //printf("\nSingle Block\n");
        numblocks = 1;
        if ( array_size < 2 ) numthreads = 2;
        else if ( array_size < 4 ) numthreads = 4;
        else if ( array_size < 8 ) numthreads = 8;
        else if ( array_size < 16 ) numthreads = 16;
        else if ( array_size < 32 ) numthreads = 32;
        else if ( array_size < 64 ) numthreads = 64;
        else if ( array_size < 128 ) numthreads = 128;
        else if ( array_size < 256 ) numthreads = 256;
        else if ( array_size < 512 ) numthreads = 512;
        else if ( array_size < 1024 ) numthreads = 1024;
    }

//    printf("Blocks: %d\n", numblocks);
//    printf("Threads: %d\n", numthreads);
//    printf("Levels: %f\n", ceil((double)numblocks / numthreads));
//    printf("Free: %zu\n", free_mem);
//    printf("Total: %zu\n", total_mem);

//    int version = 0;
//    cudaRuntimeGetVersion(&version);
//    printf("CUDA Version: %d\n\n", version);

    // MEAN ------------------------------------------------------------------------------------------------------------

    // The kernel will always return 56 sums which we
    // will add on the cpu because the overhead of adding them again on the GPU will be higher than just
    // adding them sequentially
    k_sum_blocked <<< MAX_BLOCKS, numthreads, numthreads * sizeof(double) * 3 >>> (dA, dB, device_array_size);
    checkCUDAError("k_sum_blocked", true);

    // wait for the gpu to finish
    cudaDeviceSynchronize();

    // get the result of the array sum and add it to our sum variable
    cudaMemcpy(B, dB, MAX_BLOCKS * sizeof(local_mms), cudaMemcpyDeviceToHost);
    //printf("[");
    for (int i = 0; i < MAX_BLOCKS; ++i) {
//        printf("(");
//        printf("%4.6f", B[i].sum);
//        printf("), ");
        sum += B[i].sum;
        B[i].sum = 0.0;
        if ( min > B[i].min ) min = B[i].min;
        if ( max < B[i].max ) max = B[i].max;
    }
//    printf("]\n");
//    printf("BLOCKED SUM: [%4.6f]\n", sum);

    mean = sum / array_size;
    sum = 0;

    // STD_DEV ---------------------------------------------------------------------------------------------------------

    HANDLE_ERROR( cudaMalloc( (void **)&dM, sizeof( double ) ) );
    HANDLE_ERROR( cudaMemcpy( dM, &mean, sizeof( double ), cudaMemcpyHostToDevice) );

    k_sum_stddev<<< MAX_BLOCKS, numthreads, numthreads * sizeof(double) >>> (dA, dB, device_array_size, dM);
    checkCUDAError("k_sum_blocked", true);

    // wait for the gpu to finish
    cudaDeviceSynchronize();

    cudaMemcpy(B, dB, MAX_BLOCKS * sizeof(local_mms), cudaMemcpyDeviceToHost);
    //printf("[");
    for (int i = 0; i < MAX_BLOCKS; ++i) {
//        printf("(");
//        printf("%4.6f", B[i].sum);
//        printf("), ");
        sum += B[i].sum;
    }

    stddev = sqrt( sum / ( array_size ) );


    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dM);
    cudaFree(device_array_size);
    // -----------------------------------------------------------------------------------------------------------------
    // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - END
    // -----------------------------------------------------------------------------------------------------------------

    //
    // Record the end time.
    //
    gettimeofday( &end, NULL );

    //
    // Calculate the runtime.
    //
    runtime = ( ( end.tv_sec  - start.tv_sec ) * 1000.0 ) + ( ( end.tv_usec - start.tv_usec ) / 1000.0 );

    //
    // Output discoveries from the array.
    //


    printf( "Statistics for array ( %d, %d ):\n", array_size, seed );
    printf( "    Minimum = %4.6f, Maximum = %4.6f\n", min, max );
    printf( "    Mean = %4.6f, Standard Deviation = %4.6f\n", mean, stddev );
    printf( "Processing Time: %4.4f milliseconds\n", runtime );

    //
    // Free the allocated array.
    //
    free( array );
    free( a_sum );

    return 0;
}
