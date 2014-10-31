#include "ProjHelperFun.h"
#include "Constants.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

__global__ void
updateParams_kernel(const REAL alpha, const REAL beta, const REAL nu, REAL *myVarX, REAL *myVarY, REAL *myX, REAL *myY, int numX, int numY)
{
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  myVarX[gidJ * numX + gidI] = exp(2.0 * (beta  * log(myX[gidI]) + myY[gidJ] + nu));
  myVarY[gidJ * numX + gidI] = exp(2.0 * (alpha * log(myX[gidI]) + myY[gidJ] + nu));

}

__global__ void
setPayoff_kernel(REAL* myX, REAL* myResult, unsigned int numX, unsigned int numY, unsigned int numO)
{
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidI >= numX || gidJ >= numY || gidO >= numO)
    return;

  REAL strike = 0.001*gidO;
  REAL payoff = MAX(myX[gidI] - strike, (REAL)0.0);
  myResult[(gidO * numX + gidI) * numY + gidJ] = payoff;

}

__global__ void
rollback_kernel_1(REAL *a, REAL *b, REAL *c, REAL *u, REAL *v, REAL *myResult, REAL *myVarX, REAL *myVarY, REAL *myDxx, REAL *myDyy, REAL dtInv, int numX, int numY)
{
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  const unsigned int bidI = blockIdx.x*blockDim.x;
  const unsigned int bidJ = blockIdx.y*blockDim.y;
  const unsigned int lidI = threadIdx.x;
  const unsigned int lidJ = threadIdx.y;
  const unsigned int gidI = bidI + lidI;
  const unsigned int gidJ = bidJ + lidJ;
  REAL tmp;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(bidI + lidJ >= numX || bidJ + lidI >= numY) {
    sh_mem[32*lidI + lidJ] = 0.0;
  } else {
    sh_mem[32*lidI + lidJ] = myResult[(gidO * numX + bidI + lidJ) * numY + bidJ + lidI];
  }

  if(gidI >= numX || gidJ >= numY) {
    return;
  }

  __syncthreads();

  tmp = 0.0;
  if (lidJ > 0) {
    tmp += myDyy[0 * numY + gidJ] * sh_mem[32*(lidJ-1) + lidI];
  } else if (gidJ > 0) {
    tmp += myDyy[0 * numY + gidJ] * myResult[(gidO * numX + gidI) * numY + gidJ - 1];
  }
  tmp   += myDyy[1 * numY + gidJ] * sh_mem[32*lidJ + lidI];
  if (lidJ < 31) {
    tmp += myDyy[2 * numY + gidJ] * sh_mem[32*(lidJ+1) + lidI];
  } else if (gidJ < numY-1) {
    tmp += myDyy[2 * numY + gidJ] * myResult[(gidO * numX + gidI) * numY + gidJ + 1];
  }

  REAL v_tmp = v[(gidO * numY + gidJ) * numX + gidI] = 0.5*myVarY[gidJ * numX + gidI] * tmp;

  tmp = 0.0;
  if (lidI > 0) {
    tmp += myDxx[0 * numX + gidI] * sh_mem[32*lidJ + lidI - 1];
  } else if (gidI > 0) {
    tmp += myDxx[0 * numX + gidI] * myResult[(gidO * numX + gidI - 1) * numY + gidJ];
  }
  tmp   += myDxx[1 * numX + gidI] * sh_mem[32*lidJ + lidI];
  if (lidI < 31) {
    tmp += myDxx[2 * numX + gidI] * sh_mem[32*lidJ + lidI + 1];
  } else if(gidI < numX-1) {
    tmp += myDxx[2 * numX + gidI] * myResult[(gidO * numX + gidI + 1) * numY + gidJ];
  }

  u[(gidO * numY + gidJ) * numX + gidI] = 0.5 * 0.5 * myVarX[gidJ * numX + gidI] * tmp +
    v_tmp +
    dtInv * sh_mem[32*lidJ + lidI];

  if(gidO == 0) {
    a[gidJ * numX + gidI]  =       - 0.5 * 0.5 * myVarX[gidJ * numX + gidI] * myDxx[0 * numX + gidI];
    b[gidJ * numX + gidI]  = dtInv - 0.5 * 0.5 * myVarX[gidJ * numX + gidI] * myDxx[1 * numX + gidI];
    c[gidJ * numX + gidI]  =       - 0.5 * 0.5 * myVarX[gidJ * numX + gidI] * myDxx[2 * numX + gidI];
  }
}

__global__ void
tridag_kernel_0(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x + 1;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidJ >= numY)
    return;

  yy[gidJ * numX + gidI] = -a[gidJ * numX + gidI] * c[gidJ * numX + gidI-1];
}

__global__ void
tridag_kernel_1(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  int i;

  if(gidJ >= numY)
    return;

  yy[gidJ * numX] = 1.0 / b[gidJ * numX];

  for(i = 1; i < numX; i++) {
    yy[gidJ * numX + i] = 1.0 / (b[gidJ * numX + i] + yy[gidJ * numX + i] * yy[gidJ * numX + i - 1]);
  }
}

__global__ void
tridag_kernel_2(REAL *a, REAL *b, REAL *c, REAL *u, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidI >= numX)
    return;

  if(gidO == 0) {
    if(gidI > 0) {
      a[gidJ * numX + gidI] = 1.0 /
        (c[gidJ * numX + gidI - 1] *
         yy[gidJ * numX + gidI - 1] -
         b[gidJ * numX + gidI] /
         a[gidJ * numX + gidI]
         );
    }
    b[gidJ * numX + gidI] = - c[gidJ * numX + gidI] * yy[gidJ * numX + gidI];
  }

  u[(gidO * numY + gidJ) * numX + gidI] =   u[(gidO * numY + gidJ) * numX + gidI] * yy[gidJ * numX + gidI];
}

__global__ void
tridag_kernel_3(REAL *u, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int i;

  if(gidJ >= numY)
    return;

  for(i = 1; i < numX; i++) {
    u[(gidO * numY + gidJ) * numX + i] += a[gidJ * numX + i] * u[(gidO * numY + gidJ) * numX + i - 1];
  }
  for(i = numX-2; i >= 0; i--) {
    u[(gidO * numY + gidJ) * numX + i] += b[gidJ * numX + i] * u[(gidO * numY + gidJ) * numX + i + 1];
  }
}

__global__ void
rollback_kernel_2(REAL *a, REAL *b, REAL *c, REAL *y, REAL *u, REAL *v, REAL *myDyy, REAL *myVarY, REAL dtInv, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidJ >= numY)
    return;

  if(gidO == 0) {
    a[gidI * numY + gidJ] =       - 0.25 * myVarY[gidJ * numX + gidI] * myDyy[0 * numY + gidJ];
    b[gidI * numY + gidJ] = dtInv - 0.25 * myVarY[gidJ * numX + gidI] * myDyy[1 * numY + gidJ];
    c[gidI * numY + gidJ] =       - 0.25 * myVarY[gidJ * numX + gidI] * myDyy[2 * numY + gidJ];
  }
  y[(gidO * numX + gidI) * numY + gidJ] = dtInv * u[(gidO * numY + gidJ) * numX + gidI] - 0.5 * v[(gidO * numY + gidJ) * numX + gidI];

}

__global__ void
tridag_kernel_4(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidJ >= numY)
    return;

  if(gidJ > 0) {
    yy[gidI * numY + gidJ] = -a[gidI * numY + gidJ] * c[gidI * numY + gidJ - 1];
  }
}

__global__ void
tridag_kernel_5(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  int j;

  if(gidI >= numX)
    return;

  yy[gidI * numY] = 1.0 / b[gidI * numY];

  for(j = 1; j < numY; j++) {
    yy[gidI * numY + j] = 1.0 / (b[gidI * numY + j] + yy[gidI * numY + j] * yy[gidI * numY + j - 1]);
  }
}

__global__ void
tridag_kernel_6(REAL *a, REAL *b, REAL *c, REAL *y, REAL *yy, REAL *myResult, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidJ >= numY)
    return;

  if(gidO == 0) {
    if(gidJ > 0) {
      a[gidI * numY + gidJ] = 1.0 /
        (c[gidI * numY + gidJ - 1] *
         yy[gidI * numY + gidJ - 1] -
         b[gidI * numY + gidJ] /
         a[gidI * numY + gidJ]);
    }
    b[gidI * numY + gidJ] = -c[gidI * numY + gidJ] * yy[gidI * numY + gidJ];
  }

  myResult[(gidO * numX + gidI) * numY + gidJ] = y[(gidO * numX + gidI) * numY + gidJ] * yy[gidI * numY + gidJ];
}

__global__ void
tridag_kernel_7(REAL *myResult, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int j;

  if(gidI >= numX)
    return;

  for(j = 1; j < numY; j++) {
    myResult[(gidO * numX + gidI) * numY + j] += a[gidI * numY + j] * myResult[(gidO * numX + gidI) * numY + j - 1];
  }
  for(j = numY-2; j >= 0; j--) {
    myResult[(gidO * numX + gidI) * numY + j] += b[gidI * numY + j] * myResult[(gidO * numX + gidI) * numY + j + 1];
  }
}

void
rollback(const REAL dtInv, PrivGlobs &globs)
{
  /* v[o][j][i] = myResult[o][i][j-1..j+1] & myVarY[i][j]
     u[o][j][i] = myResult[o][i-1..i+1][j] & myVarX[j][i]
     a[j][i]    = myVarX[j][i]             & myDxx[0][i]
     b[j][i]    = myVarX[j][i]             & myDxx[1][i]
     c[j][i]    = myVarX[j][i]             & myDxx[2][i]
   */
  rollback_kernel_1
    <<<
    dim3(DIVUP(globs.numX, 32), DIVUP(globs.numY, 32), globs.numO),
    dim3(32, 32, 1),
    32*32*sizeof(REAL)
    >>>
    (globs.a, globs.b, globs.c, globs.u, globs.v, globs.myResult, globs.myVarX, globs.myVarY, globs.myDxx, globs.myDyy, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* yy[j][i] = a[j][i] & c[j][i-1] */
  tridag_kernel_0
    <<<
    dim3(globs.numX-1, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* yy[j] = b[0..j] */
  tridag_kernel_1
    <<<
    dim3(1, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs.yy, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* a = func(c, yy, b, a)
     b = func(c, yy)
     u = func(u, yy) */
  tridag_kernel_2
    <<<
    dim3(DIVUP(globs.numX, 64), globs.numY, globs.numO),
    dim3(64, 1, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.u, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* u = func(u[i-1], a)
     u = func(u[i+1], b) */
  tridag_kernel_3
    <<<
    dim3(1, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.u, globs.a, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* a = func(myVarY, myDyy)
     b = func(myVarY, myDyy)
     c = func(myVarY, myDyy)
     y = func(dtInv, u, v) */
  rollback_kernel_2
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.y, globs.u, globs.v, globs.myDyy, globs.myVarY, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_4
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_5
    <<<
    dim3(DIVUP(globs.numX, 32), 1, 1),
    dim3(32, 1, 1)
    >>>
    (globs.yy, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_6
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.y, globs.yy, globs.myResult, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_7
    <<<
    dim3(DIVUP(globs.numX, 32), 1, globs.numO),
    dim3(32, 1, 1)
    >>>
    (globs.myResult, globs.a, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
}

void
value(PrivGlobs &globs,
      const REAL s0,
      const REAL t,
      const REAL alpha,
      const REAL nu,
      const REAL beta,
      REAL *res)
{

  setPayoff_kernel
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.myX, globs.myResult, globs.numX, globs.numY, globs.numO);

  for(int i = globs.numT-2; i >= 0; i--) {
    updateParams_kernel
      <<<
      dim3(globs.numX, DIVUP(globs.numY, 32), 1),
      dim3(1, 32, 1)
      >>>
      (alpha, beta, -0.5 * nu * nu * globs.myTimeline[i], globs.myVarX, globs.myVarY, globs.myX, globs.myY, globs.numX, globs.numY);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaThreadSynchronize());

    rollback(1.0 / (globs.myTimeline[i+1] - globs.myTimeline[i]), globs);
  }

  for(int i = 0; i < globs.numO; i++) {
    cudaMemcpy(&res[i], &globs.myResult[(i * globs.numX + globs.myXindex)*globs.numY + globs.myYindex], sizeof(REAL), cudaMemcpyDeviceToHost);
  }
}

void
run_OrigCPU(const unsigned int   outer,
            const unsigned int   numX,
            const unsigned int   numY,
            const unsigned int   numT,
            const REAL           s0,
            const REAL           t,
            const REAL           alpha,
            const REAL           nu,
            const REAL           beta,
            REAL*                res)   // [outer] RESULT
{
  PrivGlobs globs(numX, numY, numT, outer);
  initGrid(s0, alpha, nu, t, numX, numY, numT, outer, globs);
  initOperator(globs.myX, numX, globs.myDxx);
  initOperator(globs.myY, numY, globs.myDyy);

  value(globs, s0,   t,
        alpha, nu,   beta,
        res);
}
