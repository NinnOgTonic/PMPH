#include "ProjHelperFun.h"
#include "Constants.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DIVUP(x, y) (((x) + (y) - 1) / (y))

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
       REAL *b,   // size [n]
       const REAL *c,   // size [n]
       const REAL *r,   // size [n]
       const int             n,
       REAL *u,   // size [n]
       REAL *uu)  // size [n] temporary
{
  int i;

  // Map
  for(i=1; i<n; i++) {
    uu[i] = -a[i] * c[i-1];
  }

  // CPU-scan
  uu[0] = 1.0 / b[0];
  for(i = 1; i < n; i++) {
    uu[i] = 1.0 / (b[i] + uu[i] * uu[i-1]);
  }

  // Map
  for(i = 1; i < n; i++) {
    a[i] = 1.0 / (c[i-1] * uu[i-1] - b[i] / a[i]);
  }

  // Map
  for(i = 0; i < n-1; i++) {
    b[i] = - c[i] * uu[i];
  }

  // Map
  for(i = 0; i < n; i++) {
    u[i] = r[i] * uu[i];
  }

  // CPU-scan
  for(i=1; i<n; i++) {
    u[i] += a[i] * u[i-1];
  }

  // CPU-scan
  for(i = n-2; i >= 0; i--) {
    u[i] += b[i] * u[i+1];
  }
}


__global__ void
rollback_kernel_1(unsigned int numX, unsigned int numY, REAL *d_u, PrivGlobs &globs) {
  REAL tmp;
  const unsigned int gidJ = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidI = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int lidJ = threadIdx.x;
  const unsigned int lidI = threadIdx.y;

  extern __shared__ char sh_mem[];

  if(gidI >= numX || gidJ >= numY)
    return;

  sh_mem[lidJ + 1] = globs.myResult[gidI*globs.numY + gidJ];
  if(lidJ == 0 && gidJ > 0)
    sh_mem[lidJ] = globs.myResult[gidI*globs.numY + gidJ - 1];
  if(lidJ == blockDim.x - 1 && gidJ < numY-1)
    sh_mem[lidJ + 2] = globs.myResult[gidI*globs.numY + gidJ + 1];


  tmp = globs.myDyy[4*gidJ+1]*sh_mem[gidI*globs.numY + gidJ + 1];

  if(gidJ > 0) {
    tmp += globs.myDyy[4*gidJ+0]*sh_mem[gidI*globs.numY + gidJ + 1 -1];
  }

  if(gidJ < numY-1) {
    tmp += globs.myDyy[4*gidJ+2]*sh_mem[gidI*globs.numY + gidJ + 1 +1];
  }

  d_u[gidI*numY + gidJ] = 0.5*globs.myVarY[gidI*globs.numY+gidJ]*tmp;
}

__global__ void
rollback_kernel_2(unsigned int numX, unsigned int numY, REAL *d_u, REAL *d_v, PrivGlobs &globs) {
  REAL tmp;
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  extern __shared__ char sh_mem[];

  if(gidJ >= numX || gidI >= numY)
    return;

  sh_mem[lidJ + 1] = globs.myResult[gidI*globs.numY + gidJ];
  if(lidJ == 0 && gidJ > 0)
    sh_mem[lidJ] = globs.myResult[gidI*globs.numY + gidJ - 1];
  if(lidJ == blockDim.x - 1 && gidJ < numY-1)
    sh_mem[lidJ + 2] = globs.myResult[gidI*globs.numY + gidJ + 1];

  tmp = globs.myDxx[4*gidI+1] * globs.myResult[    gidI*globs.numY + gidJ];

  if(gidI > 0) {
    tmp += globs.myDxx[4*gidI+0] * globs.myResult[(gidI-1)*globs.numY + gidJ];
  }

  if(gidI < numX-1) {
    tmp += globs.myDxx[4*gidI+2] * globs.myResult[(gidI+1)*globs.numY + gidJ];
  }

  d_u[gidJ*numX + gidI] = dtInv*globs.myResult[gidI*globs.numY+gidJ] + 0.25*globs.myVarX[gidI*globs.numY+gidJ]*tmp + d_v[gidI*numY+gidJ];
}

void
rollback(const unsigned g, PrivGlobs& globs)
{
  const unsigned blockSizeX = 64;
  const unsigned blockSizeY =  1;
  const dim3 grid  (DIVUP(totalWidth, blockSizeX), DIVUP(totalHeight, blockSizeY), 1);
  const dim3 block (blockSizeX, blockSizeY, 1);
  const unsigned shared = blockSizeX + 2;

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
  REAL *d_v, d_u;

  cudaMalloc((REAL**)&d_v, numY*numX*sizeof(REAL));
  cudaMalloc((REAL**)&d_u, numY*numX*sizeof(REAL));

  rollback_kernel_1<<< grid, block, shared >>>(numX, numY, d_v, globs);
  cudaMemcpy(v, d_v, numY*numX*sizeof(REAL), cudaMemcpyDeviceToHost);
  // map
  /*
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
  */

  rollback_kernel_2<<< grid, block, shared >>>(numX, numY, d_v, d_u, globs);
  cudaMemcpy(u, d_u, numY*numX*sizeof(REAL), cudaMemcpyDeviceToHost);
  /*
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
  */

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
#pragma omp parallel for
  for(unsigned i = 0; i < outer; i++) {
    res[i] = value(s0,    0.001*i, t,
                   alpha, nu,      beta,
                   numX,  numY,    numT );
  }
}

//#endif // PROJ_CORE_ORIG
