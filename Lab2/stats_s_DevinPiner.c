#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define  MAXIMUM_VALUE   1000000

int main( int argc, char* argv[] ) {
    //
    // Determine min, max, mean and standard deviation of array
    //
    unsigned int array_size, seed, i;
    struct timeval start, end;
    float runtime;

    if( argc < 3 ) {
        printf( "Format: stats_s <size of array> <random seed>\n" );
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
    // Record the start time.
    //
    gettimeofday( &start, NULL );

    //
    // Allocate the array to be populated.
    //
    double *array = (double *) malloc( array_size * sizeof( double ) );

    //
    // Seed the random number generator and populate the array with its values.
    //
    srand( seed );
    for( i = 0; i < array_size; i++ )
    array[i] = ( (double) rand() / (double) RAND_MAX ) * MAXIMUM_VALUE;

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

    //
    // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - BEGIN
    //
    for ( i = 0; i < array_size; ++i ) {
        sum += array[i];
        if ( array[i] > max ) max = array[i];
        if ( array[i] < min ) min = array[i];
    }

    mean = sum / array_size;

    double stddev_sum = 0;
    for ( i = 0; i < array_size; ++i ) {
        double dif = array[i] -  mean;
        stddev_sum += dif * dif;
    }

    stddev = sqrt(stddev_sum / (array_size));
    //
    // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - END
    //

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

    return 0;
}
