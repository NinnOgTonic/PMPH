#include "ProjHelperFun.h"
#include "Constants.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

unsigned long long total_count1, total_count2;

static
int timeval_subtract1(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff<0);
}

__global__ void
updateParams_kernel(const REAL alpha, const REAL beta, const REAL nu, REAL *myVarX, REAL *myVarY, REAL *myX, REAL *myY, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  myVarX[gidI * numY + gidJ] =  exp(2.0 * (beta  * log(myX[gidI]) + myY[gidJ] + nu));
  myVarY[gidI * numY + gidJ] =  exp(2.0 * (alpha * log(myX[gidI]) + myY[gidJ] + nu));

}

__global__ void
setPayoff_kernel(REAL strike, REAL* myX, REAL* myResult, unsigned int numX, unsigned int numY){
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  REAL payoff = MAX(myX[gidI] - strike, (REAL)0.0);
  myResult[gidI * numY + gidJ] = payoff;

}

__global__ void
rollback_kernel_1(REAL *v, REAL *myResult, REAL *myVarY, REAL *myDyy, int numX, int numY) {
  const unsigned int gidI = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidJ = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int lidJ = threadIdx.x;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  sh_mem[lidJ + 1] = myResult[gidI * numY + gidJ];
  if(lidJ == 0) {
    if (gidJ == 0) {
      sh_mem[lidJ] = 0;
    } else {
      sh_mem[lidJ] = myResult[gidI * numY + gidJ - 1];
    }
  }

  if(gidJ == numY-1) {
    sh_mem[lidJ + 2] = 0;
  } else if(lidJ == blockDim.x - 1) {
    sh_mem[lidJ + 2] = myResult[gidI * numY + gidJ + 1];
  }

  __syncthreads();

  v[gidI*numY + gidJ] = 0.5*myVarY[gidI*numY+gidJ] *
    (myDyy[0 * numY + gidJ] * sh_mem[lidJ] +
     myDyy[1 * numY + gidJ] * sh_mem[lidJ + 1] +
     myDyy[2 * numY + gidJ] * sh_mem[lidJ + 2]);
}

__global__ void
rollback_kernel_2(REAL *v, REAL *u, REAL *myResult, REAL *myVarX, REAL *myDxx, REAL dtInv, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int lidI = threadIdx.x;
  const unsigned int lidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  sh_mem[34*lidJ + lidI + 1] = myResult[gidI * numY + gidJ];
  if(lidI == 0) {
    if (gidI == 0) {
      sh_mem[34*lidJ + lidI] = 0;
    } else {
      sh_mem[34*lidJ + lidI] = myResult[(gidI - 1) * numY + gidJ];
    }
  }

  if(gidI == numX-1) {
    sh_mem[34*lidJ + lidI + 2] = 0;
  } else if(lidI == blockDim.x - 1) {
    sh_mem[34*lidJ + lidI + 2] = myResult[(gidI + 1) * numY + gidJ];
  }

  __syncthreads();

  u[gidJ*numX + gidI] = 0.25*myVarX[gidI*numY+gidJ] *
    (myDxx[0 * numX + gidI] * sh_mem[34*lidJ + lidI] +
     myDxx[1 * numX + gidI] * sh_mem[34*lidJ + lidI + 1] +
     myDxx[2 * numX + gidI] * sh_mem[34*lidJ + lidI + 2]) +
    v[gidI*numY+gidJ] +
    dtInv * myResult[gidI*numY+gidJ];
}

__global__ void
rollback_kernel_3(REAL *a, REAL *b, REAL *c, REAL *myVarX, REAL *myDxx, REAL dtInv, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int tidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  if (tidJ < 3) {
    sh_mem[tidJ] = -0.25 * myDxx[tidJ * numX + gidI];
  }

  __syncthreads();

  a[gidJ * numX + gidI]  =         myVarX[gidI*numY+gidJ]*sh_mem[0];
  b[gidJ * numX + gidI]  = dtInv + myVarX[gidI*numY+gidJ]*sh_mem[1];
  c[gidJ * numX + gidI]  =         myVarX[gidI*numY+gidJ]*sh_mem[2];
}

__global__ void
rollback_kernel_4(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x + 1;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  yy[gidJ * numX + gidI] = -a[gidJ * numX + gidI] * c[gidJ * numX + gidI-1];
}

__global__ void
rollback_kernel_5(REAL *a, REAL *b, REAL *c, REAL *y, REAL *u, REAL *v, REAL *yy, REAL *myDyy, REAL *myVarY, REAL dtInv, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int tidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  if (tidJ < 3) {
    sh_mem[tidJ] = -0.25 * myDyy[tidJ * numY + gidJ];
  }

  __syncthreads();

  a[gidI * numY + gidJ] =         myVarY[gidI*numY + gidJ] * sh_mem[0];
  b[gidI * numY + gidJ] = dtInv + myVarY[gidI*numY + gidJ] * sh_mem[1];
  c[gidI * numY + gidJ] =         myVarY[gidI*numY + gidJ] * sh_mem[2];
  y[gidI * numY + gidJ] = dtInv * u[gidJ * numX + gidI] - 0.5 * v[gidI * numY + gidJ];

}

__global__ void
rollback_kernel_6(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  if(gidJ > 0) {
    yy[gidI * numY + gidJ] = -a[gidI * numY + gidJ] * c[gidI * numY + gidJ-1];
  }
}

__global__ void
tridag_kernel_1(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  int i;

  if(gidJ >= numY)
    return;

  REAL last = yy[gidJ*numX] = 1.0 / b[gidJ*numX];

  for(i = 1; i < numX; i++) {
    last = yy[gidJ*numX+i] = 1.0 / (b[gidJ*numX+i] + yy[gidJ*numX+i] * last);
  }
}

__global__ void
tridag_kernel_2(REAL *a, REAL *b, REAL *c, REAL *u, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  if(gidI > 0) {
    a[gidJ*numX+gidI] = 1.0 / (c[gidJ*numX+gidI-1] * yy[gidJ*numX+gidI-1] - b[gidJ*numX+gidI] / a[gidJ*numX+gidI]);
  }

  REAL cur_yy = yy[gidJ*numX+gidI];
  b[gidJ*numX+gidI] = - c[gidJ*numX+gidI] * cur_yy;
  u[gidJ*numX+gidI] =   u[gidJ*numX+gidI] * cur_yy;
}

__global__ void
tridag_kernel_3(REAL *u, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  int i;

  if(gidJ >= numY)
    return;

  REAL last = u[gidJ*numX];

  for(i = 1; i < numX; i++) {
    last = u[gidJ*numX+i] = u[gidJ*numX+i] + a[gidJ*numX+i] * last;
  }
  for(i = numX-2; i >= 0; i--) {
    last = u[gidJ*numX+i] = u[gidJ*numX+i] + b[gidJ*numX+i] * last;
  }
}

__global__ void
tridag_kernel_4(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  int j;

  if(gidI >= numX)
    return;

  REAL last = yy[gidI*numY] = 1.0 / b[gidI*numY];

  for(j = 1; j < numY; j++) {
    last = yy[gidI*numY+j] = 1.0 / (b[gidI*numY+j] + yy[gidI*numY+j] * last);
  }
}

__global__ void
tridag_kernel_5(REAL *a, REAL *b, REAL *c, REAL *y, REAL *yy, REAL *results, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  if(gidJ > 0) {
    a[gidI*numY+gidJ] = 1.0 / (c[gidI*numY+gidJ-1] * yy[gidI*numY+gidJ-1] - b[gidI*numY+gidJ] / a[gidI*numY+gidJ]);
  }
  b[gidI*numY+gidJ]       = - c[gidI*numY+gidJ] * yy[gidI*numY+gidJ];
  results[gidI*numY+gidJ] =   y[gidI*numY+gidJ] * yy[gidI*numY+gidJ];
}

__global__ void
tridag_kernel_6(REAL *results, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  int j;

  if(gidI >= numX)
    return;

  REAL last = results[gidI*numY];

  for(j = 1; j < numY; j++) {
    last = results[gidI*numY+j] = results[gidI*numY+j] + a[gidI*numY+j] * last;
  }
  for(j = numY-2; j >= 0; j--) {
    last = results[gidI*numY+j] = results[gidI*numY+j] + b[gidI*numY+j] * last;
  }
}

void
rollback(const REAL dtInv, PrivGlobs &globs)
{

  /* v = func(myResult, myVarY, myDyy) */
  rollback_kernel_1
    <<<
    dim3(DIVUP(globs.numY, 1024), globs.numX, 1),
    dim3(1024, 1, 1),
    sizeof(REAL) * (1024 + 2)
    >>>
    (globs.v, globs.myResult, globs.myVarY, globs.myDyy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());


  {
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    /* u = func(myResult, myVarX, myDxx, dtInv, v) */
    rollback_kernel_2
      <<<
      dim3(DIVUP(globs.numX, 32), DIVUP(globs.numY, 32), 1),
      dim3(32, 32, 1),
      sizeof(REAL) * (32 + 2) * 32
      >>>
      (globs.v, globs.u, globs.myResult, globs.myVarX, globs.myDxx, dtInv, globs.numX, globs.numY);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaThreadSynchronize());

    gettimeofday(&t_end, NULL);
    timeval_subtract1(&t_diff, &t_end, &t_start);
    total_count2 += t_diff.tv_sec*1e6+t_diff.tv_usec;
  }

  /* a = func(myVarX, myDxx)
     b = func(myVarX, myDxx, dtInv)
     c = func(myVarX, myDxx) */
  rollback_kernel_3
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (globs.a, globs.b, globs.c, globs.myVarX, globs.myDxx, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  /* yy = func(a, c) */
  rollback_kernel_4
    <<<
    dim3(globs.numX-1, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  {
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    /* yy = func(b, yy[i-1]) */
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
      dim3(DIVUP(globs.numX, 32), globs.numY, 1),
      dim3(32, 1, 1)
      >>>
      (globs.a, globs.b, globs.c, globs.u, globs.yy, globs.numX, globs.numY);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaThreadSynchronize());

    /* u = func(u[i-1], a)
       u = func(u[i+1], b) */
    tridag_kernel_3
      <<<
      dim3(1, DIVUP(globs.numY, 32), 1),
      dim3(1, 32, 1)
      >>>
      (globs.u, globs.a, globs.b, globs.numX, globs.numY);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaThreadSynchronize());

    gettimeofday(&t_end, NULL);
    timeval_subtract1(&t_diff, &t_end, &t_start);
    total_count1 += t_diff.tv_sec*1e6+t_diff.tv_usec;
  }

  rollback_kernel_5
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (globs.a, globs.b, globs.c, globs.y, globs.u, globs.v, globs.yy, globs.myDyy, globs.myVarY, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_6
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_4
    <<<
    dim3(DIVUP(globs.numX, 32), 1, 1),
    dim3(32, 1, 1)
    >>>
    (globs.yy, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_5
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.y, globs.yy, globs.myResult, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_6
    <<<
    dim3(DIVUP(globs.numX, 32), 1, 1),
    dim3(32, 1, 1)
    >>>
    (globs.myResult, globs.a, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
}

REAL
value(PrivGlobs &globs,
      const REAL s0,
      const REAL strike,
      const REAL t,
      const REAL alpha,
      const REAL nu,
      const REAL beta,
      const unsigned int numX,
      const unsigned int numY,
      const unsigned int numT)
{
  REAL res;

  setPayoff_kernel
    <<<
    dim3(numX, DIVUP(numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (strike, globs.myX, globs.myResult, numX, numY);

  for(int i = numT-2; i >= 0; i--) {
    updateParams_kernel
      <<<
      dim3(numX, DIVUP(numY, 32), 1),
      dim3(1, 32, 1)
      >>>
      (alpha, beta, -0.5 * nu * nu * globs.myTimeline[i], globs.myVarX, globs.myVarY, globs.myX, globs.myY, numX, numY);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaThreadSynchronize());

    rollback(1.0 / (globs.myTimeline[i+1] - globs.myTimeline[i]), globs);
  }

  cudaMemcpy(&res, &globs.myResult[globs.myXindex*globs.numY+globs.myYindex], sizeof(REAL), cudaMemcpyDeviceToHost);

  return res;
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
  total_count1 = 0;
  total_count2 = 0;

  PrivGlobs globs(numX, numY, numT);
  initGrid(s0, alpha, nu, t, numX, numY, numT, &globs);
  initOperator(globs.myX, numX, globs.myDxx);
  initOperator(globs.myY, numY, globs.myDyy);

  for(unsigned i = 0; i < outer; i++) {
    PrivGlobs clone = globs.cudaClone();
    res[i] = value(clone,
                   s0,    0.001*i, t,
                   alpha, nu,      beta,
                   numX,  numY,    numT);
  }
  printf("%lld %lld\n", total_count1, total_count2);
}
