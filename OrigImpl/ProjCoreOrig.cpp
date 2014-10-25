#include "ProjHelperFun.h"
#include "Constants.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void
updateParams(const unsigned g,
             const REAL alpha,
             const REAL beta,
             const REAL nu,
             PrivGlobs& globs)
{
  for(unsigned i = 0; i < globs.numX; i++)
    for(unsigned j = 0; j < globs.numY; j++) {
      globs.myVarX[i*globs.numY+j] = exp(2.0*(
                                              beta*log(globs.myX[i])
                                              + globs.myY[j]
                                              - 0.5*nu*nu*globs.myTimeline[g] )
                                         );
      globs.myVarY[i*globs.numY+j] = exp(2.0*(
                                              alpha*log(globs.myX[i])
                                              + globs.myY[j]
                                              - 0.5*nu*nu*globs.myTimeline[g]
                                              )
                                         ); // nu*nu
    }
}

void
setPayoff(const REAL strike, PrivGlobs& globs)
{
  for(unsigned i=0; i < globs.numX; i++)
    {
      REAL payoff = MAX(globs.myX[i]-strike, (REAL)0.0);
      for(unsigned j=0; j<globs.numY; j++)
        globs.myResult[i*globs.numY+j] = payoff;
    }
}

inline void
tridag(REAL *a,   // size [n]
       const REAL *b,   // size [n]
       REAL *c,   // size [n]
       const REAL *r,   // size [n]
       const int             n,
       REAL *u,   // size [n]
       REAL *uu)  // size [n] temporary
{
  int    i, offset;
  REAL   beta;

  u[0]  = r[0];
  uu[0] = b[0];

  // Some kind of scan
  for(i=1; i<n; i++) {
    beta  = a[i] / uu[i-1];
    uu[i] = b[i] - beta*c[i-1];
    u[i]  = r[i] - beta*u[i-1];
  }

  // Map
  for(i = 0; i < n; i++) {
    u[i] = u[i] / uu[i];
    c[i] = c[i] / uu[i];
  }

  // Scan
  for(i=n-2; i>=0; i--) {
    u[i] -= c[i]*u[i+1];
  }
}

void
rollback(const unsigned g, PrivGlobs& globs)
{
  unsigned
    numX = globs.numX,
    numY = globs.numY;

  unsigned numZ = MAX(numX,numY);

  unsigned i, j;

  REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

  REAL *u  = (REAL*) malloc(numY*numX*sizeof(REAL));
  REAL *v  = (REAL*) malloc(numY*numX*sizeof(REAL));
  REAL *a  = (REAL*) malloc(numZ*sizeof(REAL));
  REAL *b  = (REAL*) malloc(numZ*sizeof(REAL));
  REAL *c  = (REAL*) malloc(numZ*sizeof(REAL));
  REAL *y  = (REAL*) malloc(numZ*sizeof(REAL));
  REAL *yy = (REAL*) malloc(numZ*sizeof(REAL));


  // map
  for(i = 0; i < numX; i++) {
    for(j = 0; j < numY; j++) {
      REAL tmp = globs.myDyy[4*j+1]*globs.myResult[i*globs.numY + j];

      if(j > 0) {
        tmp += globs.myDyy[4*j+0]*globs.myResult[i*globs.numY + j-1];
      }

      if(j < numY-1) {
        tmp += globs.myDyy[4*j+2]*globs.myResult[i*globs.numY + j+1];
      }

      v[i*numY + j] = 0.5*globs.myVarY[i*globs.numY+j]*tmp;
    }
  }

  // map
  for(j = 0; j < numY; j++) {
    // stencil
    for(i = 0; i < numX; i++) {
      REAL tmp = globs.myDxx[4*i+1] * globs.myResult[    i*globs.numY + j];

      if(i > 0) {
        tmp += globs.myDxx[4*i+0] * globs.myResult[(i-1)*globs.numY + j];
      }

      if(i < numX-1) {
        tmp += globs.myDxx[4*i+2] * globs.myResult[(i+1)*globs.numY + j];
      }

      u[j*numX + i] = dtInv*globs.myResult[i*globs.numY+j] + 0.25*globs.myVarX[i*globs.numY+j]*tmp + v[i*numY+j];
    }
  }

  //  implicit x
  //  Vi skriver hver iteration til index i, så derfor skal de privatiseres
  //  og derefter kan vi unrolle dem så de er flade
  for(j = 0; j < numY; j++) {
    // map
    for(i = 0; i < numX; i++) {  // here a, b,c should have size [numX]
      a[i] =       - 0.5*0.5*globs.myVarX[i*globs.numY+j]*globs.myDxx[4*i + 0];
      b[i] = dtInv - 0.5*0.5*globs.myVarX[i*globs.numY+j]*globs.myDxx[4*i + 1];
      c[i] =       - 0.5*0.5*globs.myVarX[i*globs.numY+j]*globs.myDxx[4*i + 2];
    }
    // here yy should have size [numX]
    tridag(a,b,c,&u[j*numX],numX,&u[j*numX],yy);
  }

  for(i = 0; i < numX; i++) {
    for(j = 0; j < numY; j++) {  // here a, b, c should have size [numY]
      a[j] =       - 0.5*0.5*globs.myVarY[i*globs.numY+j]*globs.myDyy[4*j + 0];
      b[j] = dtInv - 0.5*0.5*globs.myVarY[i*globs.numY+j]*globs.myDyy[4*j + 1];
      c[j] =       - 0.5*0.5*globs.myVarY[i*globs.numY+j]*globs.myDyy[4*j + 2];
      y[j] = dtInv * u[j*numX+i] - 0.5*v[i*numY+j];
    }

    // here yy should have size [numY]
    tridag(a,b,c,y,numY,&globs.myResult[i*numY],yy);
  }
  free(u);
  free(v);
  free(a);
  free(b);
  free(c);
  free(y);
  free(yy);
}

REAL
value(const REAL s0,
      const REAL strike,
      const REAL t,
      const REAL alpha,
      const REAL nu,
      const REAL beta,
      const unsigned int numX,
      const unsigned int numY,
      const unsigned int numT)
{
  PrivGlobs    globs(numX, numY, numT);
  initGrid(s0, alpha, nu, t, numX, numY, numT, globs);
  initOperator(globs.myX, numX, globs.myDxx);
  initOperator(globs.myY, numY, globs.myDyy);

  setPayoff(strike, globs);

  for(int i = globs.numT-2; i >= 0; i--) {
    updateParams(i,alpha,beta,nu,globs);
    rollback(i, globs);
  }

  return globs.myResult[globs.myXindex*globs.numY+globs.myYindex];
}

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
            REAL*                 res)   // [outer] RESULT
{
  printf("%d %d %d\n", numX, numY, numT);
#pragma omp parallel for
  for(unsigned i = 0; i < outer; i++) {
    res[i] = value(s0,    0.001*i, t,
                   alpha, nu,      beta,
                   numX,  numY,    numT );
  }
}

//#endif // PROJ_CORE_ORIG
