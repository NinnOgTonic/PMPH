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

inline void
tridag(REAL *a,
       REAL *b,
       const REAL *c,
       const REAL *r,
       const int n,
       REAL *u,
       REAL *uu)
{
  int i;

  // Map
  /* This was originally part of tridag, but was moved outside it */
  // for(i=1; i<n; i++) {
  //   uu[i] = -a[i] * c[i-1];
  // }

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
rollback_kernel_1(PrivGlobs *globs, REAL *v) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int lidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= globs->numX || gidJ >= globs->numY)
    return;

  sh_mem[lidJ + 1] = globs->myResult[gidI * globs->numY + gidJ];
  if(lidJ == 0) {
    if (gidJ == 0) {
      sh_mem[lidJ] = 0;
    } else {
      sh_mem[lidJ] = globs->myResult[gidI * globs->numY + gidJ - 1];
    }
  }

  if(gidJ == globs->numY-1) {
    sh_mem[lidJ + 2] = 0;
  } else if(lidJ == blockDim.y - 1) {
    sh_mem[lidJ + 2] = globs->myResult[gidI * globs->numY + gidJ + 1];
  }

  __syncthreads();

  v[gidI*globs->numY + gidJ] = 0.5*globs->myVarY[gidI*globs->numY+gidJ] *
    (globs->myDyy[4*gidJ+0] * sh_mem[lidJ] +
     globs->myDyy[4*gidJ+1] * sh_mem[lidJ + 1] +
     globs->myDyy[4*gidJ+2] * sh_mem[lidJ + 2]);
}

__global__ void
rollback_kernel_2(PrivGlobs *globs, REAL *v, REAL *u, REAL dtInv) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int lidI = threadIdx.x;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= globs->numX || gidJ >= globs->numY)
    return;

  sh_mem[lidI + 1] = globs->myResult[gidI * globs->numY + gidJ];
  if(lidI == 0) {
    if (gidI == 0) {
      sh_mem[lidI] = 0;
    } else {
      sh_mem[lidI] = globs->myResult[(gidI - 1) * globs->numY + gidJ];
    }
  }

  if(gidI == globs->numX-1) {
    sh_mem[lidI + 2] = 0;
  } else if(lidI == blockDim.x - 1) {
    sh_mem[lidI + 2] = globs->myResult[(gidI + 1) * globs->numY + gidJ];
  }

  __syncthreads();

  u[gidJ*globs->numX + gidI] = 0.25*globs->myVarX[gidI*globs->numY+gidJ] *
    (globs->myDxx[4*gidI+0] * sh_mem[lidI] +
     globs->myDxx[4*gidI+1] * sh_mem[lidI + 1] +
     globs->myDxx[4*gidI+2] * sh_mem[lidI + 2]) +
    v[gidI*globs->numY+gidJ] +
    dtInv * globs->myResult[gidI*globs->numY+gidJ];
}

__global__ void
rollback_kernel_3(PrivGlobs *globs, REAL *a, REAL *b, REAL *c, REAL dtInv) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int tidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= globs->numX || gidJ >= globs->numY)
    return;

  if (tidJ < 3) {
    sh_mem[tidJ] = -0.25 * globs->myDxx[4*gidI + tidJ];
  }

  __syncthreads();

  a[gidJ * globs->numX + gidI]  =         globs->myVarX[gidI*globs->numY+gidJ]*sh_mem[0];
  b[gidJ * globs->numX + gidI]  = dtInv + globs->myVarX[gidI*globs->numY+gidJ]*sh_mem[1];
  c[gidJ * globs->numX + gidI]  =         globs->myVarX[gidI*globs->numY+gidJ]*sh_mem[2];
}

__global__ void
rollback_kernel_4(PrivGlobs *globs, REAL *a, REAL *c, REAL *yy) {
  // Kernel for the yy part.
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x + 1;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= globs->numX || gidJ >= globs->numY)
    return;

  yy[gidJ * globs->numX + gidI] = -a[gidJ * globs->numX + gidI] * c[gidJ * globs->numX + gidI-1];
}

__global__ void
rollback_kernel_5(PrivGlobs  * globs, REAL *a, REAL *b, REAL *c, REAL *y, REAL *u, REAL *v, REAL *yy, REAL dtInv) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int tidJ = threadIdx.y;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem = (REAL*) sh_mem1;

  if(gidI >= globs->numX || gidJ >= globs->numY)
    return;

  if (tidJ < 3) {
    sh_mem[tidJ] = -0.25 * globs->myDyy[4*gidJ + tidJ];
  }

  __syncthreads();

  a[gidI * globs->numY + gidJ] =         globs->myVarY[gidI * globs->numY+gidJ] * sh_mem[0];
  b[gidI * globs->numY + gidJ] = dtInv + globs->myVarY[gidI * globs->numY+gidJ] * sh_mem[1];
  c[gidI * globs->numY + gidJ] =         globs->myVarY[gidI * globs->numY+gidJ] * sh_mem[2];
  y[gidI * globs->numY + gidJ] = dtInv * u[gidJ * globs->numX + gidI] - 0.5 * v[gidI * globs->numY + gidJ];

}

__global__ void
rollback_kernel_6(PrivGlobs *globs, REAL *a, REAL *c, REAL *yy) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= globs->numX || gidJ >= globs->numY)
    return;

  if(gidJ > 0) {
    yy[gidI * globs->numY + gidJ] = -a[gidI * globs->numY + gidJ] * c[gidI * globs->numY + gidJ-1];
  }
}

__global__ void
tridag_kernel_1(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  int i;

  if(gidJ >= numY)
    return;

  yy[gidJ*numX] = 1.0 / b[gidJ*numX];

  for(i = 1; i < numX; i++) {
    yy[gidJ*numX+i] = 1.0 / (b[gidJ*numX+i] + yy[gidJ*numX+i] * yy[gidJ*numX+i-1]);
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
  b[gidJ*numX+gidI] = - c[gidJ*numX+gidI] * yy[gidJ*numX+gidI];
  u[gidJ*numX+gidI] =   u[gidJ*numX+gidI] * yy[gidJ*numX+gidI];
}

void
rollback(const unsigned g, PrivGlobs *globs)
{
  int
    numX = globs->numX,
    numY = globs->numY;

  int i, j;

  REAL dtInv = 1.0/(globs->myTimeline[g+1]-globs->myTimeline[g]);

  REAL *u;
  REAL *v;
  REAL *a;
  REAL *b;
  REAL *c;
  REAL *y;
  REAL *yy;

  checkCudaError(cudaMallocHost(&u,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMallocHost(&v,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMallocHost(&a,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMallocHost(&b,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMallocHost(&c,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMallocHost(&y,  numY*numX*sizeof(REAL)));
  checkCudaError(cudaMallocHost(&yy, numY*numX*sizeof(REAL)));

  rollback_kernel_1
    <<<
    dim3(globs->numX, DIVUP(globs->numY, 64), 1),
    dim3(1, 64, 1),
    sizeof(REAL) * (64 + 2)
    >>>
    (globs, v);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_2
    <<<
    dim3(DIVUP(globs->numX, 64), globs->numY, 1),
    dim3(64, 1, 1),
    sizeof(REAL) * (64 + 2)
    >>>
    (globs, v, u, dtInv);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_3
    <<<
    dim3(globs->numX, DIVUP(globs->numY, 32), 1),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (globs, a, b, c, dtInv);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  rollback_kernel_4
    <<<
    dim3(globs->numX-1, DIVUP(globs->numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs, a, c, yy);
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

  for(j = 0; j < numY; j++) {
    for(i = 1; i < numX; i++) {
      u[j*numX+i] += a[j*numX+i] * u[j*numX+i-1];
    }
    for(i = numX-2; i >= 0; i--) {
      u[j*numX+i] += b[j*numX+i] * u[j*numX+i+1];
    }
  }

  rollback_kernel_5
    <<<
    dim3(globs->numX, DIVUP(globs->numY, 32), 1),
    dim3(1, 32, 1),
    sizeof(REAL) * 3
    >>>
    (globs, a, b, c, y, u, v, yy, dtInv);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  //for(i = 0; i < numX; i++) {
  //  for(j = 0; j < numY; j++) {
  //    a[i*numY + j] =       - 0.5*0.5*globs->myVarY[i*globs->numY+j]*globs->myDyy[4*j + 0];
  //    b[i*numY + j] = dtInv - 0.5*0.5*globs->myVarY[i*globs->numY+j]*globs->myDyy[4*j + 1];
  //    c[i*numY + j] =       - 0.5*0.5*globs->myVarY[i*globs->numY+j]*globs->myDyy[4*j + 2];
  //    y[i*numY + j] = dtInv * u[j*numX+i] - 0.5*v[i*numY+j];
  //    /*if(j > 0) {
  //      yy[i*numY + j] = -a[i*numY + j] * c[i*numY + j-1];
  //    }*/
  //  }
  //}

  rollback_kernel_6
    <<<
    dim3(globs->numX, DIVUP(globs->numY, 32), 1),
    dim3(1, 32, 1)
    >>>
    (globs, a, c, yy);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());

  for(i = 0; i < numX; i++) {
    yy[i*numY] = 1.0 / b[i*numY];
    for(j = 1; j < numY; j++) {
      yy[i*numY+j] = 1.0 / (b[i*numY+j] + yy[i*numY+j] * yy[i*numY+j-1]);
    }
  }

  for(i = 0; i < numX; i++) {
    for(j = 1; j < numY; j++) {
      a[i*numY+j] = 1.0 / (c[i*numY+j-1] * yy[i*numY+j-1] - b[i*numY+j] / a[i*numY+j]);
    }

    // Map
    for(j = 0; j < numY-1; j++) {
      b[i*numY+j] = - c[i*numY+j] * yy[i*numY+j];
    }

    // Map
    for(j = 0; j < numY; j++) {
      globs->myResult[i*numY+j] = y[i*numY+j] * yy[i*numY+j];
    }
  }

  for(i = 0; i < numX; i++) {
    for(j = 1; j < numY; j++) {
      globs->myResult[i*numY+j] += a[i*numY+j] * globs->myResult[i*numY+j-1];
    }

    // CPU-scan
    for(j = numY - 2; j >= 0; j--) {
      globs->myResult[i*numY+j] += b[i*numY+j] * globs->myResult[i*numY+j+1];
    }
  }
  checkCudaError(cudaFreeHost(u));
  checkCudaError(cudaFreeHost(v));
  checkCudaError(cudaFreeHost(a));
  checkCudaError(cudaFreeHost(b));
  checkCudaError(cudaFreeHost(c));
  checkCudaError(cudaFreeHost(y));
  checkCudaError(cudaFreeHost(yy));
}

REAL
value(PrivGlobs *globs,
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
  setPayoff(strike, globs);

  for(int i = numT-2; i >= 0; i--) {
    updateParams(i, alpha, beta, nu, globs);
    rollback(i, globs);
  }

  return globs->myResult[globs->myXindex*globs->numY+globs->myYindex];
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

  for(unsigned i = 0; i < outer; i++) {
    PrivGlobs globs(numX, numY, numT);
    initGrid(s0, alpha, nu, t, numX, numY, numT, &globs);
    initOperator(globs.myX, numX, globs.myDxx);
    initOperator(globs.myY, numY, globs.myDyy);

    PrivGlobs *clone = globs.clone();
    res[i] = value(clone,
                   s0,    0.001*i, t,
                   alpha, nu,      beta,
                   numX,  numY,    numT);
    clone->~PrivGlobs();
    checkCudaError(cudaFreeHost(clone));
  }
}
