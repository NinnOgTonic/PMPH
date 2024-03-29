#ifndef PARSE_INPUT
#define PARSE_INPUT

#include "ParserC.h"
#include <assert.h>
#include <math.h>
#include <algorithm>

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
//#include <sys/timeb.h>
#include <assert.h>


using namespace std;

#include <iostream>
using std::cout;
using std::endl;

#include "Constants.h"

#define MAX_VAL 1024

#if WITH_FLOATS
    #define read_real read_float
#else
    #define read_real read_double
#endif

const float EPS = 0.00001;


/*******************************************************/
/*****  Utilities Related to Time Instrumentation  *****/
/*******************************************************/

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

bool is_pow2(int atr_val) {
    int x = 1;

    for(int i = 0; i < 31; i++) {
        if(x == atr_val) return true;
        x = (x << 1);
    }
    return false;
}


/***********************************/
/********** READ DATA SET **********/
/***********************************/

void readDataSet(   unsigned int& outer,
                    unsigned int& num_X,
                    unsigned int& num_Y,
                    unsigned int& num_T
) {
    if(     read_int( static_cast<unsigned int*>( &outer ) ) ||
            read_int( static_cast<unsigned int*>( &num_X ) ) ||
            read_int( static_cast<unsigned int*>( &num_Y ) ) ||
            read_int( static_cast<unsigned int*>( &num_T ) )  ) {

        fprintf(stderr, "Syntax error when reading the dataset, i.e., four ints.\n");
        exit(1);
    }

    {   // check dataset invariants:
        bool atr_ok = true;

        atr_ok  = outer > 0;
        assert(atr_ok && "Outer loop count less than 0!");

        atr_ok  = (num_X > 0) && (num_X <= MAX_VAL);
        assert(atr_ok && "Illegal NUM_X value!");

        atr_ok  = (num_Y > 0) && (num_Y <= MAX_VAL);
        assert(atr_ok && "Illegal NUM_X value!");

        atr_ok  = num_T > 0;
        assert(atr_ok && "NUM_T value less or equal to zero!!");
    }
}

REAL* readOutput( const int& N ) {
    REAL*   result;
    int64_t shape[3];

    // reading the standard output
    if (read_array(sizeof(REAL), read_real, (void**)&result, shape, 1) ) {
        fprintf(stderr, "Syntax error when reading the output.\n");
        exit(1);
    }

    bool ok = ( shape[0] == N );
    assert(ok && "Incorrect shape of the standard result!");

    return result;
}

bool validate( const REAL* res, const int& N ) {
    bool  is_valid = true;

    REAL* std_res = readOutput( N );

    for ( int i = 0; i < N; i ++ ) {
        float err = fabs(std_res[i] - res[i]);
        if ( isnan(res[i]) || isinf(res[i]) || err > EPS ) {
            is_valid = false;
            fprintf(stderr, "Error[%d] = %f, EPS = %f!\n", i, err, EPS);
            break;
        }
    }

    return is_valid;
}

void writeStatsAndResult(   const bool& valid, const REAL* data,
                            const int & outer, const int & num_X,
                            const int & num_Y, const int & num_T,
                            const bool& is_gpu,const int & P,
                            const unsigned long int& elapsed
) {
    // print stats to stdout
    fprintf(stdout, "// OUTER=%d, NUM_X=%d, NUM_Y=%d, NUM_T=%d.\n",
                    outer, num_X, num_Y, num_T     );

    if(valid) { fprintf(stdout, "1\t\t// VALID   Result,\n"); }
    else      { fprintf(stdout, "0\t\t// INVALID Result,\n"); }

    fprintf(stdout, "%ld\t\t// Runtime in microseconds,\n", elapsed);
    if(is_gpu) fprintf(stdout, "%d\t\t// GPU Threads,\n\n", P);
    else       fprintf(stdout, "%d\t\t// CPU Threads,\n\n", P);

    // write the result
    write_1Darr( data, static_cast<int>(outer),
                       "PMPH Project Result"  );
}

#if 0
void writeResult( const REAL* res, const unsigned int N, const char* msg ) {
    //ofstream fout;
    //fout.open ("output.data");
    cout << "\n[ ";
    for( int k=0; k<N-1; ++k ) {
        cout<<res[k]<<", "<<endl;
    }
    cout << res[N-1] << " ]\t// "<<msg<< endl;

    //fout.close();
}
#endif
#endif // PARSE_INPUT
