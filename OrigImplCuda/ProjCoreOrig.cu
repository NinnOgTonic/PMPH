#include "ProjHelperFun.h"
#include "Constants.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DIVUP(x, y) (((x) + (y) - 1) / (y))

void
updateParams(const unsigned g,
             const REAL alpha,
             const REAL beta,
             const REAL nu,
             PrivGlobs *globs)
{
  for(unsigned i = 0; i < globs->numX; i++)
    for(unsigned j = 0; j < globs->numY; j++) {
      globs->myVarX[i*globs->numY+j] = exp(2.0*(
                                                beta*log(globs->myX[i])
                                                + globs->myY[j]
                                                - 0.5*nu*nu*globs->myTimeline[g] )
                                           );
      globs->myVarY[i*globs->numY+j] = exp(2.0*(
                                                alpha*log(globs->myX[i])
                                                + globs->myY[j]
                                                - 0.5*nu*nu*globs->myTimeline[g]
                                                )
                                           ); // nu*nu
    }
}


__global__ void
updateParams_kernel(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, REAL *myVarX, REAL *myVarY, REAL *myX, REAL *myY, REAL *myTimeline, unsigned int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x + 1;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  myVarX[gidI * numY + gidJ] =  exp(2.0 * (beta * log(myX[gidI]) + myY[gidJ] - 0.5 * nu * nu * myTimeline[g])); 
  myVarY[gidI * numY + gidJ] =  exp(2.0 * (alpha * log(myX[gidI]) + myY[gidJ] - 0.5 * nu * nu * myTimeline[g])); 

}

void
setPayoff(const REAL strike, PrivGlobs *globs)
{
  for(unsigned i=0; i < globs->numX; i++)
    {
      REAL payoff = MAX(globs->myX[i]-strike, (REAL)0.0);
      for(unsigned j=0; j<globs->numY; j++)
        globs->myResult[i*globs->numY+j] = payoff;
    }
}

__global__ void
rollback_kernel_1(REAL *v, REAL *myResult, REAL *myVarY, REAL *myDyy, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int lidJ = threadIdx.y;

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
  } else if(lidJ == blockDim.y - 1) {
    sh_mem[lidJ + 2] = myResult[gidI * numY + gidJ + 1];
  }

  __syncthreads();

  v[gidI*numY + gidJ] = 0.5*myVarY[gidI*numY+gidJ] *
    (myDyy[4*gidJ+0] * sh_mem[lidJ] +
     myDyy[4*gidJ+1] * sh_mem[lidJ + 1] +
     myDyy[4*gidJ+2] * sh_mem[lidJ + 2]);
}

__global__ void
rollback_kernel_2(REAL *v, REAL *u, REAL *myResult, REAL *myVarX, REAL *myDxx, REAL dtInv, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int lidI = threadIdx.x;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= numX || gidJ >= numY)
    return;

  sh_mem[lidI + 1] = myResult[gidI * numY + gidJ];
  if(lidI == 0) {
    if (gidI == 0) {
      sh_mem[lidI] = 0;
    } else {
      sh_mem[lidI] = myResult[(gidI - 1) * numY + gidJ];
    }
  }

  if(gidI == numX-1) {
    sh_mem[lidI + 2] = 0;
  } else if(lidI == blockDim.x - 1) {
    sh_mem[lidI + 2] = myResult[(gidI + 1) * numY + gidJ];
  }

  __syncthreads();

  u[gidJ*numX + gidI] = 0.25*myVarX[gidI*numY+gidJ] *
    (myDxx[4*gidI+0] * sh_mem[lidI] +
     myDxx[4*gidI+1] * sh_mem[lidI + 1] +
     myDxx[4*gidI+2] * sh_mem[lidI + 2]) +
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
    sh_mem[tidJ] = -0.25 * myDxx[4*gidI + tidJ];
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
    sh_mem[tidJ] = -0.25 * myDyy[4*gidJ + tidJ];
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
rollback(const unsigned g, PrivGlobs *globs)
{
  int
    numX = globs->numX,
    numY = globs->numY;

  REAL dtInv = 1.0/(globs->myTimeline[g+1]-globs->myTimeline[g]);

  REAL *u;
  REAL *v;
  REAL *a;
  REAL *b;
  REAL *c;
  REAL *y;
  REAL *yy;

  checkCudaError(cudaMalloc(&u,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMalloc(&v,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMalloc(&a,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMalloc(&b,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMalloc(&c,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMalloc(&y,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMalloc(&yy, numY*numX*sizeof(REAL)));

  rollback_kernel_1
    <<<
    dim3(numX, DIVUP(numY, 64), 1),
    dim3(1, 64, 1),
    sizeof(REAL) * (64 + 2)
    >>>
    (v, globs->myResult, globs->myVarY, globs->myDyy, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_2
    <<<
    dim3(DIVUP(numX, 64), numY, 1),
    dim3(64, 1, 1),
    sizeof(REAL) * (64 + 2)
    >>>
    (v, u, globs->myResult, globs->myVarX, globs->myDxx, dtInv, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_3
    <<<
    dim3(numX, DIVUP(numY, 32), 1),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (a, b, c, globs->myVarX, globs->myDxx, dtInv, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_4
    <<<
    dim3(numX-1, DIVUP(numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (a, c, yy, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_1
    <<<
    dim3(1, DIVUP(numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (yy, b, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_2
    <<<
    dim3(DIVUP(numX, 32), numY, 1),
    dim3(32, 1, 1)
    >>>
    (a, b, c, u, yy, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_3
    <<<
    dim3(1, DIVUP(numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (u, a, b, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_5
    <<<
    dim3(numX, DIVUP(numY, 32), 1),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (a, b, c, y, u, v, yy, globs->myDyy, globs->myVarY, dtInv, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_6
    <<<
    dim3(numX, DIVUP(numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (a, c, yy, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_4
    <<<
    dim3(DIVUP(numX, 32), 1, 1),
    dim3(32, 1, 1)
    >>>
    (yy, b, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_5
    <<<
    dim3(globs->numX, DIVUP(globs->numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (a, b, c, y, yy, globs->myResult, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  tridag_kernel_6
    <<<
    dim3(DIVUP(numX, 32), 1, 1),
    dim3(32, 1, 1)
    >>>
    (globs->myResult, a, b, numX, numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  checkCudaError(cudaFree(u));
  checkCudaError(cudaFree(v));
  checkCudaError(cudaFree(a));
  checkCudaError(cudaFree(b));
  checkCudaError(cudaFree(c));
  checkCudaError(cudaFree(y));
  checkCudaError(cudaFree(yy));
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
  setPayoff(strike, &globs);

  for(int i = numT-2; i >= 0; i--) {
    //updateParams(i, alpha, beta, nu, &globs);
    updateParams_kernel
    <<<
    dim3(numX, DIVUP(numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (i, alpha, beta, nu, globs.myVarX, globs.myVarY, globs.myX, globs.myY, globs.myTimeline, numY);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaThreadSynchronize());

    rollback(i, &globs);
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

  PrivGlobs globs(numX, numY, numT, true);
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
}
