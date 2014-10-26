#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <new>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Constants.h"

using namespace std;

static inline void
checkCudaError(cudaError err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
    fflush(stderr);
    exit(1);
  }
}


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

  PrivGlobs( ) {
    printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
    exit(0);
  }

  PrivGlobs(const unsigned int& numX,
            const unsigned int& numY,
            const unsigned int& numT,
            const bool useCuda)
  {
    this->numX = numX;
    this->numY = numY;
    this->numT = numT;
    if(useCuda) {
      checkCudaError(cudaMallocHost(&this->myX,        numX * sizeof(REAL)));
      checkCudaError(cudaMallocHost(&this->myDxx,      numX * sizeof(REAL) * 4));
      checkCudaError(cudaMallocHost(&this->myY,        numY * sizeof(REAL)));
      checkCudaError(cudaMallocHost(&this->myDyy,      numY * sizeof(REAL) * 4));
      checkCudaError(cudaMallocHost(&this->myTimeline, numT * sizeof(REAL)));
      checkCudaError(cudaMallocHost(&this->myVarX,     numX * numY * sizeof(REAL)));
      checkCudaError(cudaMallocHost(&this->myVarY,     numX * numY * sizeof(REAL)));
      checkCudaError(cudaMallocHost(&this->myResult,   numX * numY * sizeof(REAL)));
    } else {
      this->myX        = (REAL*) malloc(numX * sizeof(REAL));
      this->myDxx      = (REAL*) malloc(numX * sizeof(REAL) * 4);
      this->myY        = (REAL*) malloc(numY * sizeof(REAL));
      this->myDyy      = (REAL*) malloc(numY * sizeof(REAL) * 4);
      this->myTimeline = (REAL*) malloc(numT * sizeof(REAL));
      this->myVarX     = (REAL*) malloc(numX * numY * sizeof(REAL));
      this->myVarY     = (REAL*) malloc(numX * numY * sizeof(REAL));
      this->myResult   = (REAL*) malloc(numX * numY * sizeof(REAL));
    }
  }

  PrivGlobs cudaClone() {
    PrivGlobs other(this->numX, this->numY, this->numT, true);

    memcpy(other.myX,         this->myX,        numX * sizeof(REAL));
    memcpy(other.myDxx,       this->myDxx,      numX * sizeof(REAL) * 4);
    memcpy(other.myY,         this->myY,        numY * sizeof(REAL));
    memcpy(other.myDyy,       this->myDyy,      numY * sizeof(REAL) * 4);
    memcpy(other.myTimeline,  this->myTimeline, numT * sizeof(REAL));
    memcpy(other.myVarX,      this->myVarX,     numX * numY * sizeof(REAL));
    memcpy(other.myVarY,      this->myVarY,     numX * numY * sizeof(REAL));
    memcpy(other.myResult,    this->myResult,   numX * numY * sizeof(REAL));

    other.myXindex = this->myXindex;
    other.myYindex = this->myYindex;

    return other;
  }

  ~PrivGlobs() {
    checkCudaError(cudaFreeHost(this->myX));
    checkCudaError(cudaFreeHost(this->myDxx));
    checkCudaError(cudaFreeHost(this->myY));
    checkCudaError(cudaFreeHost(this->myDyy));
    checkCudaError(cudaFreeHost(this->myTimeline));
    checkCudaError(cudaFreeHost(this->myVarX));
    checkCudaError(cudaFreeHost(this->myVarY));
    checkCudaError(cudaFreeHost(this->myResult));
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
