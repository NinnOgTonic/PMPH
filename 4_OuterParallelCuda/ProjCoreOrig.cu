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

  myVarX[gidI * numY + gidJ] =  exp(2.0 * (beta  * log(myX[gidI]) + myY[gidJ] + nu));
  myVarY[gidI * numY + gidJ] =  exp(2.0 * (alpha * log(myX[gidI]) + myY[gidJ] + nu));

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
rollback_kernel_1(REAL *v, REAL *myResult, REAL *myVarY, REAL *myDyy, int numX, int numY)
{
  const unsigned int gidI = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidJ = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  const unsigned int lidJ = threadIdx.x;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  sh_mem[lidJ + 1] = myResult[(gidO * numX + gidI) * numY + gidJ];
  if(lidJ == 0) {
    if (gidJ == 0) {
      sh_mem[lidJ] = 0;
    } else {
      sh_mem[lidJ] = myResult[(gidO * numX + gidI) * numY + gidJ - 1];
    }
  }

  if(gidJ == numY-1) {
    sh_mem[lidJ + 2] = 0;
  } else if(lidJ == blockDim.x - 1) {
    sh_mem[lidJ + 2] = myResult[(gidO * numX + gidI) * numY + gidJ + 1];
  }

  __syncthreads();

  v[(gidO * numX + gidI) * numY + gidJ] = 0.5*myVarY[gidI*numY+gidJ] *
    (myDyy[0 * numY + gidJ] * sh_mem[lidJ] +
     myDyy[1 * numY + gidJ] * sh_mem[lidJ + 1] +
     myDyy[2 * numY + gidJ] * sh_mem[lidJ + 2]);
}

__global__ void
rollback_kernel_2(REAL *v, REAL *u, REAL *myResult, REAL *myVarX, REAL *myDxx, REAL dtInv, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  const unsigned int lidI = threadIdx.x;
  const unsigned int lidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  sh_mem[34*lidJ + lidI + 1] = myResult[(gidO * numX + gidI) * numY + gidJ];
  if(lidI == 0) {
    if (gidI == 0) {
      sh_mem[34*lidJ + lidI] = 0;
    } else {
      sh_mem[34*lidJ + lidI] = myResult[(gidO * numX + (gidI - 1)) * numY + gidJ];
    }
  }

  if(gidI == numX-1) {
    sh_mem[34*lidJ + lidI + 2] = 0;
  } else if(lidI == blockDim.x - 1) {
    sh_mem[34*lidJ + lidI + 2] = myResult[(gidO * numX + (gidI + 1)) * numY + gidJ];
  }

  __syncthreads();

  u[(gidO * numY + gidJ) * numX + gidI] = 0.25*myVarX[ gidI * numY + gidJ] *
    (myDxx[0 * numX + gidI] * sh_mem[34 * lidJ + lidI] +
     myDxx[1 * numX + gidI] * sh_mem[34 * lidJ + lidI + 1] +
     myDxx[2 * numX + gidI] * sh_mem[34 * lidJ + lidI + 2]) +
    v[(gidO * numX + gidI) * numY + gidJ] +
    dtInv * myResult[(gidO * numX + gidI) * numY + gidJ];
}

__global__ void
rollback_kernel_3(REAL *a, REAL *b, REAL *c, REAL *myVarX, REAL *myDxx, REAL dtInv, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  const unsigned int tidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  if (tidJ < 3) {
    sh_mem[tidJ] = -0.25 * myDxx[tidJ * numX + gidI];
  }

  __syncthreads();

  a[(gidO * numY + gidJ) * numX + gidI]  =         myVarX[gidI * numY + gidJ] * sh_mem[0];
  b[(gidO * numY + gidJ) * numX + gidI]  = dtInv + myVarX[gidI * numY + gidJ] * sh_mem[1];
  c[(gidO * numY + gidJ) * numX + gidI]  =         myVarX[gidI * numY + gidJ] * sh_mem[2];
}

__global__ void
tridag_kernel_0(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x + 1;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidI >= numX || gidJ >= numY)
    return;

  yy[(gidO * numY + gidJ) * numX + gidI] = -a[(gidO * numY + gidJ) * numX + gidI] * c[(gidO * numY + gidJ) * numX + gidI-1];
}

__global__ void
tridag_kernel_1(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int i;

  if(gidJ >= numY)
    return;

  yy[(gidO * numY + gidJ) * numX] = 1.0 / b[(gidO * numY + gidJ) * numX];

  for(i = 1; i < numX; i++) {
    yy[(gidO * numY + gidJ) * numX + i] = 1.0 / (b[(gidO * numY + gidJ) * numX + i] + yy[(gidO * numY + gidJ) * numX + i] * yy[(gidO * numY + gidJ) * numX + i - 1]);
  }
}

__global__ void
tridag_kernel_2(REAL *a, REAL *b, REAL *c, REAL *u, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidI >= numX || gidJ >= numY)
    return;

  if(gidI > 0) {
    a[(gidO * numY + gidJ) * numX + gidI] = 1.0 / (c[(gidO * numY + gidJ) * numX + gidI - 1] * yy[(gidO * numY + gidJ) * numX + gidI - 1] - b[(gidO * numY + gidJ) * numX + gidI] / a[(gidO * numY + gidJ) * numX + gidI]);
  }

  b[(gidO * numY + gidJ) * numX + gidI] = - c[(gidO * numY + gidJ) * numX + gidI] * yy[(gidO * numY + gidJ) * numX + gidI];
  u[(gidO * numY + gidJ) * numX + gidI] =   u[(gidO * numY + gidJ) * numX + gidI] * yy[(gidO * numY + gidJ) * numX + gidI];
}

__global__ void
tridag_kernel_3(REAL *u, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int i;

  if(gidJ >= numY)
    return;

  for(i = 1; i < numX; i++) {
    u[(gidO * numY + gidJ) * numX + i] += a[(gidO * numY + gidJ) * numX + i] * u[(gidO * numY + gidJ) * numX + i - 1];
  }
  for(i = numX-2; i >= 0; i--) {
    u[(gidO * numY + gidJ) * numX + i] += b[(gidO * numY + gidJ) * numX + i] * u[(gidO * numY + gidJ) * numX + i + 1];
  }
}


__global__ void
rollback_kernel_4(REAL *a, REAL *b, REAL *c, REAL *y, REAL *u, REAL *v, REAL *yy, REAL *myDyy, REAL *myVarY, REAL dtInv, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  const unsigned int tidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  if (tidJ < 3) {
    sh_mem[tidJ] = -0.25 * myDyy[tidJ * numY + gidJ];
  }

  __syncthreads();

  a[(gidO * numX + gidI) * numY + gidJ] =         myVarY[gidI * numY + gidJ] * sh_mem[0];
  b[(gidO * numX + gidI) * numY + gidJ] = dtInv + myVarY[gidI * numY + gidJ] * sh_mem[1];
  c[(gidO * numX + gidI) * numY + gidJ] =         myVarY[gidI * numY + gidJ] * sh_mem[2];
  y[(gidO * numX + gidI) * numY + gidJ] = dtInv * u[(gidO * numY + gidJ) * numX + gidI] - 0.5 * v[(gidO * numX + gidI) * numY + gidJ];

}

__global__ void
tridag_kernel_4(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidI >= numX || gidJ >= numY)
    return;

  if(gidJ > 0) {
    yy[(gidO * numX + gidI) * numY + gidJ] = -a[(gidO * numX + gidI) * numY + gidJ] * c[(gidO * numX + gidI) * numY + gidJ - 1];
  }
}

__global__ void
tridag_kernel_5(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int j;

  if(gidI >= numX)
    return;

  yy[(gidO * numX + gidI) * numY] = 1.0 / b[(gidO * numX + gidI) * numY];

  for(j = 1; j < numY; j++) {
    yy[(gidO * numX + gidI) * numY + j] = 1.0 / (b[(gidO * numX + gidI) * numY + j] + yy[(gidO * numX + gidI) * numY + j] * yy[(gidO * numX + gidI) * numY + j - 1]);
  }
}

__global__ void
tridag_kernel_6(REAL *a, REAL *b, REAL *c, REAL *y, REAL *yy, REAL *myResult, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidI >= numX || gidJ >= numY)
    return;

  if(gidJ > 0) {
    a[(gidO * numX + gidI) * numY + gidJ] = 1.0 / (c[(gidO * numX + gidI) * numY + gidJ - 1] * yy[(gidO * numX + gidI) * numY + gidJ - 1] - b[(gidO * numX + gidI) * numY + gidJ] / a[(gidO * numX + gidI) * numY + gidJ]);
  }
  b       [(gidO * numX + gidI) * numY + gidJ] = - c[(gidO * numX + gidI) * numY + gidJ] * yy[(gidO * numX + gidI) * numY + gidJ];
  myResult[(gidO * numX + gidI) * numY + gidJ] =   y[(gidO * numX + gidI) * numY + gidJ] * yy[(gidO * numX + gidI) * numY + gidJ];
}

__global__ void
tridag_kernel_7(REAL *myResult, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int j;

  if(gidI >= numX)
    return;

  for(j = 1; j < numY; j++) {
    myResult[(gidO * numX + gidI) * numY + j] += a[(gidO * numX + gidI) * numY + j] * myResult[(gidO * numX + gidI) * numY + j - 1];
  }
  for(j = numY-2; j >= 0; j--) {
    myResult[(gidO * numX + gidI) * numY + j] += b[(gidO * numX + gidI) * numY + j] * myResult[(gidO * numX + gidI) * numY + j + 1];
  }
}

void
rollback(const REAL dtInv, PrivGlobs &globs)
{

  /* v[o][i][j] = 0.5 * myVarY[i][j]*myDyy[i][0..2] `dot` myResult[o][i][j-1..j+1] */
  rollback_kernel_1
    <<<
    dim3(DIVUP(globs.numY, 32), globs.numX, globs.numO),
    dim3(32, 1, 1),
    sizeof(REAL) * (32 + 2)
    >>>
    (globs.v, globs.myResult, globs.myVarY, globs.myDyy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* u[o][j][i] = dtInv * myResult[o][i][j] +
     0.5 * 0.5 * myVarX[i][j]*myDxx[0..2][i] `dot` myResult[o][i-1..i+1][j] +
     v[o][i][j]
  */
  rollback_kernel_2
    <<<
    dim3(DIVUP(globs.numX, 32), DIVUP(globs.numY, 32), globs.numO),
    dim3(32, 32, 1),
    sizeof(REAL) * (32 + 2) * 32
    >>>
    (globs.v, globs.u, globs.myResult, globs.myVarX, globs.myDxx, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* a[o][j][i] =       - 0.5 * 0.5 * myVarX[i][j] * myDxx[0][i]
     b[o][j][i] = dtInv - 0.5 * 0.5 * myVarX[i][j] * myDxx[1][i]
     c[o][j][i] =       - 0.5 * 0.5 * myVarX[i][j] * myDxx[2][i]
  */
  rollback_kernel_3
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (globs.a, globs.b, globs.c, globs.myVarX, globs.myDxx, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* yy[o][j][i] = -a[o][j][i] * c[o][j][i-1] */
  tridag_kernel_0
    <<<
    dim3(globs.numX-1, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* yy[o][j][0] = 1.0 / b[o][j][0]
     for i = 1..N:
       yy[o][j][i] = 1.0 / (b[o][j][i] - yy[o][j][i] * yy[o][j][i-1])
  */
  tridag_kernel_1
    <<<
    dim3(1, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.yy, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* a[o][j][i] = 1.0 / (c[o][j][i-1] * yy[o][j][i-1] - b[o][j][i] / a[o][j][i])
     b[o][j][i] = - c[o][j][i] * yy[o][j][i]
     u[o][j][i] =   u[o][j][i] * yy[o][j][i]
  */
  tridag_kernel_2
    <<<
    dim3(DIVUP(globs.numX, 32), globs.numY, globs.numO),
    dim3(32, 1, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.u, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* loop i = 1..N:
       u[o][j][i] += a[o][j][i] * u[o][j][i-1]
     loop i = N-1..0:
       u[o][j][i] += b[o][j][i] * u[o][j][i+1]
  */
  tridag_kernel_3
    <<<
    dim3(1, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.u, globs.a, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* a[o][i][j] =       - 0.5 * 0.5 * myVarY[i][j] * myDyy[0][i]
     b[o][i][j] = dtInv - 0.5 * 0.5 * myVarY[i][j] * myDyy[1][i]
     c[o][i][j] =       - 0.5 * 0.5 * myVarY[i][j] * myDyy[2][i]
     y[o][i][j] = dtInv * u[o][j][i] - 0.5 * v[o][i][j]
  */
  rollback_kernel_4
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (globs.a, globs.b, globs.c, globs.y, globs.u, globs.v, globs.yy, globs.myDyy, globs.myVarY, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* yy[o][i][j] = -a[o][i][j] * c[o][i][j-1] */
  tridag_kernel_4
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* yy[o][i][0] = 1.0 / b[o][i][0]
     for j = 1..N:
       yy[o][i][j] = 1.0 / (b[o][i][j] - yy[o][i][j] * yy[o][i][j-1])
  */
  tridag_kernel_5
    <<<
    dim3(DIVUP(globs.numX, 32), 1, globs.numO),
    dim3(32, 1, 1)
    >>>
    (globs.yy, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* a[o][i][j] = 1.0 / (c[o][i][j-1] * yy[o][i][j-1] - b[o][i][j] / a[o][i][j])
     b[o][i][j] = - c[o][i][j] * yy[o][i][j]
     myResults[o][i][j] =   y[o][i][j] * yy[o][i][j]
  */
  tridag_kernel_6
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), globs.numO),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.y, globs.yy, globs.myResult, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* loop j = 1..N:
       myResults[o][i][j] += a[o][i][j] * myResults[o][i][j-1]
     loop j = N-1..0:
       myResults[o][i][j] += b[o][i][j] * myResults[o][i][j+1]
  */
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
