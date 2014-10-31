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

#define DEBUG 1
#if DEBUG
#define cudaThreadSynchronize() cudaSuccess
#endif

struct PrivGlobs {

  //  grid
  unsigned int numX;
  unsigned int numY;
  unsigned int numT;
  REAL     *myX;        // [numX]
  REAL     *myY;        // [numY]
  REAL     *myTimeline; // [numT]
  unsigned myXindex;
  unsigned myYindex;

  //  variable
  REAL *myResult; // [numX][numY]

  //  coeffs
  REAL *myVarX; // [numX][numY]
  REAL *myVarY; // [numX][numY]

  //  operators
  REAL *myDxx;  // [numX][4]
  REAL *myDyy;  // [numY][4]

  REAL *u, *v, *a, *b, *c, *y, *yy;

  PrivGlobs( ) {
    printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
    exit(0);
  }

  PrivGlobs(const unsigned int& numX,
            const unsigned int& numY,
            const unsigned int& numT)
  {
    this->numX = numX;
    this->numY = numY;
    this->numT = numT;

    checkCudaError(cudaMalloc(&this->myX,        numX        * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myDxx,      numX * 3    * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myY,        numY        * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myDyy,      numY * 3    * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myVarX,     numX * numY * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myVarY,     numX * numY * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->myResult,   numX * numY * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->u,          numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->v,          numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->a,          numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->b,          numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->c,          numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->y,          numY * numX * sizeof(REAL)));
    checkCudaError(cudaMalloc(&this->yy,         numY * numX * sizeof(REAL)));

    this->myTimeline = (REAL*) malloc(numT * sizeof(REAL));
  }

  PrivGlobs cudaClone() {
    PrivGlobs other(this->numX, this->numY, this->numT);

    cudaMemcpy(other.myX,         this->myX,        numX        * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(other.myDxx,       this->myDxx,      numX * 3    * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(other.myY,         this->myY,        numY        * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(other.myDyy,       this->myDyy,      numY * 3    * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(other.myVarX,      this->myVarX,     numX * numY * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(other.myVarY,      this->myVarY,     numX * numY * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(other.myResult,    this->myResult,   numX * numY * sizeof(REAL), cudaMemcpyDeviceToDevice);

    memcpy(other.myTimeline,  this->myTimeline, numT        * sizeof(REAL));

    other.myXindex = this->myXindex;
    other.myYindex = this->myYindex;

    return other;
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
         const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs *globs);

void
initOperator(const REAL *x, const unsigned n, REAL *Dxx);

void
updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs *globs);

void
setPayoff(const REAL strike, PrivGlobs *globs);

void
tridag(const REAL *a,   // size [n]
       const REAL *b,   // size [n]
       const REAL *c,   // size [n]
       const REAL *r,   // size [n]
       const int  n,
       REAL       *u,   // size [n]
       REAL       *uu); // size [n] temporary

void
rollback(const unsigned g, PrivGlobs& globs);

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
run_OrigCPU(const unsigned int&   outer,
            const unsigned int&   numX,
            const unsigned int&   numY,
            const unsigned int&   numT,
            const REAL&           s0,
            const REAL&           t,
            const REAL&           alpha,
            const REAL&           nu,
            const REAL&           beta,
            REAL*                 res);   // [outer] RESULT

#endif // PROJ_HELPER_FUNS
