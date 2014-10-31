#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <new>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Constants.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DIVUP(x, y) (((x) + (y) - 1) / (y))


using namespace std;

static inline void
checkCudaError(cudaError err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
    fflush(stderr);
    exit(1);
  }
}

#define DEBUG 0
#if DEBUG
#define cudaThreadSynchronize() cudaSuccess
#endif

struct PrivGlobs {

  //  grid
  int numX;
  int numY;
  int numT;
  int numO;
  REAL     *myX;        // [numX]
  REAL     *myY;        // [numY]
  REAL     *myTimeline; // [numT]
  unsigned myXindex;
  unsigned myYindex;

  //  variable
  REAL *myResult; // [numO][numX][numY]

  //  coeffs
  REAL *myVarX; // [numX][numY]
  REAL *myVarY; // [numX][numY]

  //  operators
  REAL *myDxx;  // [3][numX]
  REAL *myDyy;  // [3][numY]

  REAL *u;  // [numO][numX][numY]
  REAL *v;  // [numO][numX][numY]
  REAL *a;  // [numY][numX]
  REAL *b;  // [numY][numX]
  REAL *c;  // [numY][numX]
  REAL *y;  // [numO][numX][numY]
  REAL *yy; // [numY][numX]

  PrivGlobs( ) {
    printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
    exit(0);
  }

  PrivGlobs(const unsigned int numX,
            const unsigned int numY,
            const unsigned int numT,
            const unsigned int numO)
  {
    this->numX = numX;
    this->numY = numY;
    this->numT = numT;
    this->numO = numO;

    checkCudaError(cudaMalloc(&this->myX,               numX        * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myDxx,             numX * 3    * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myY,               numY        * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myDyy,             numY * 3    * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myVarX,     numT * numX * numY * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myVarY,     numT * numX * numY * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myResult,   numO * numX * numY * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->u,          numO * numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->v,          numO * numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->a,                 numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->b,                 numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->c,                 numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->y,          numO * numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->yy,                numY * numX * sizeof(REAL)));

    this->myTimeline = (REAL*) malloc(numT * sizeof(REAL));
  }

  ~PrivGlobs() {
    checkCudaError(cudaFree(this->myX));
    checkCudaError(cudaFree(this->myDxx));
    checkCudaError(cudaFree(this->myY));
    checkCudaError(cudaFree(this->myDyy));
    free(this->myTimeline);
    checkCudaError(cudaFree(this->myVarX));
    checkCudaError(cudaFree(this->myVarY));
    checkCudaError(cudaFree(this->myResult));
  }
};


void
initGrid(const REAL s0, const REAL alpha, const REAL nu,const REAL t,
         const unsigned int numX, const unsigned int numY, const unsigned int numT,
         const unsigned int numO, PrivGlobs &globs);

void
initOperator(const REAL *x, const unsigned int n, REAL *Dxx);
void
rollback(const REAL g, PrivGlobs& globs);

REAL
value(PrivGlobs  &globs,
      const REAL s0,
      const REAL strike,
      const REAL t,
      const REAL alpha,
      const REAL nu,
      const REAL beta,
      const unsigned int numX,
      const unsigned int numY,
      const unsigned int numT);

void
run_OrigCPU(const unsigned int   numO,
            const unsigned int   numX,
            const unsigned int   numY,
            const unsigned int   numT,
            const REAL           s0,
            const REAL           t,
            const REAL           alpha,
            const REAL           nu,
            const REAL           beta,
            REAL*                res);   // [numO] RESULT

#endif // PROJ_HELPER_FUNS
