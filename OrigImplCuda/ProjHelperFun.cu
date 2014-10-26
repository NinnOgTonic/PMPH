#include "ProjHelperFun.h"
#include <cuda_runtime.h>

/**************************/
/**** HELPER FUNCTIONS ****/
/**************************/

/**
 * Fills in:
 *   globs.myTimeline  of size [0..numT-1]
 *   globs.myX         of size [0..numX-1]
 *   globs.myY         of size [0..numY-1]
 * and also sets
 *   globs.myXindex and globs.myYindex (both scalars)
 */
void
initGrid(const REAL s0,
         const REAL alpha,
         const REAL nu,
         const REAL t,
         const unsigned numX,
         const unsigned numY,
         const unsigned numT,
         PrivGlobs *globs)
{
  REAL *tmp = (REAL*) malloc(sizeof(REAL) * MAX(numX, numY));

  for(unsigned i = 0; i < numT; i++) {
    globs->myTimeline[i] = t*i/(numT-1);
  }

  const REAL stdX = 20.0*alpha*s0*sqrt(t);
  const REAL dx = stdX/numX;
  globs->myXindex = static_cast<unsigned>(s0/dx) % numX;


  for(unsigned i = 0; i < numX; i++) {
    tmp[i] = i*dx - globs->myXindex*dx + s0;
  }
  cudaMemcpy(globs->myX, tmp, sizeof(REAL) * numX, cudaMemcpyHostToDevice);


  const REAL stdY = 10.0*nu*sqrt(t);
  const REAL dy = stdY/numY;
  const REAL logAlpha = log(alpha);
  globs->myYindex = static_cast<unsigned>(numY/2.0);

  for(unsigned i = 0; i < numY; i++) {
    tmp[i] = i*dy - globs->myYindex*dy + logAlpha;
  }
  cudaMemcpy(globs->myY, tmp, sizeof(REAL) * numY, cudaMemcpyHostToDevice);
  free(tmp);
}

/**
 * Fills in:
 *    Dx  [0..n-1][0..3] and
 *    Dxx [0..n-1][0..3]
 * Based on the values of x,
 * Where x's size is n.
 */
__global__ void
initOperatorKernel(const REAL *x, const unsigned n, REAL *Dxx)
{
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  REAL dxl, dxu;

  if(gidI == 0 || gidI == n-1) {
    Dxx[0 * n + gidI] =  0.0;
    Dxx[1 * n + gidI] =  0.0;
    Dxx[2 * n + gidI] =  0.0;
  } else if(gidI < n-1) {
    dxl      = x[gidI]   - x[gidI-1];
    dxu      = x[gidI+1] - x[gidI];

    Dxx[0 * n + gidI] =  2.0/dxl/(dxl+dxu);
    Dxx[1 * n + gidI] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
    Dxx[2 * n + gidI] =  2.0/dxu/(dxl+dxu);
  }
}

void
initOperator(const REAL *x, const unsigned n, REAL *Dxx)
{
  //const unsigned n = x.size();

  initOperatorKernel
    <<<
    dim3(DIVUP(n, 32), 1, 1),
    dim3(32, 1, 1)
    >>>
    (x, n, Dxx);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
}
