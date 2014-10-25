#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

/* #include <vector> */
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Constants.h"

using namespace std;


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
            const unsigned int& numT)
  {
    this->numX = numX;
    this->numY = numY;
    this->numT = numT;
    this->myX =   (REAL*)malloc(numX * sizeof(REAL));
    this->myDxx = (REAL*)malloc(numX * sizeof(REAL) * 4);

    this->myY =   (REAL*)malloc(numY * sizeof(REAL));
    this->myDyy = (REAL*)malloc(numY * sizeof(REAL) * 4);

    this->myTimeline = (REAL*)malloc(numT * sizeof(REAL));

    this->myVarX   = (REAL*)malloc(numX * numY * sizeof(REAL));
    this->myVarY   = (REAL*)malloc(numX * numY * sizeof(REAL));
    this->myResult = (REAL*)malloc(numX * numY * sizeof(REAL));
  }

  PrivGlobs clone() {
    PrivGlobs clone = PrivGlobs(this->numX, this->numY, this->numT);
    /* memcpy(clone.myX,        this->myX,        sizeof(REAL) * numX); */
    /* memcpy(clone.myDxx,      this->myDxx,      sizeof(REAL) * numX * 4); */
    /* memcpy(clone.myY,        this->myY,        sizeof(REAL) * numY); */
    /* memcpy(clone.myDyy,      this->myDyy,      sizeof(REAL) * numY * 4); */
    /* memcpy(clone.myTimeline, this->myTimeline, sizeof(REAL) * numT); */
    /* memcpy(clone.myVarX,     this->myVarX,     sizeof(REAL) * numX * numY); */
    /* memcpy(clone.myVarY,     this->myVarY,     sizeof(REAL) * numX * numY); */
    /* memcpy(clone.myResult,   this->myResult,   sizeof(REAL) * numX * numY); */
    return clone;
  }

  ~PrivGlobs() {
    free(this->myX);
    free(this->myDxx);
    free(this->myY);
    free(this->myDyy);
    free(this->myTimeline);
    free(this->myVarX);
    free(this->myVarY);
    free(this->myResult);
  }
};


void
initGrid(const REAL s0, const REAL alpha, const REAL nu,const REAL t,
         const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs);

void
initOperator(const REAL *x, const unsigned n, REAL *Dxx);

void
updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void
setPayoff(const REAL strike, PrivGlobs& globs);

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
value(PrivGlobs    globs,
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
