#include "ProjHelperFun.h"
#include "Constants.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

static __global__ void
updateParams_kernel(const REAL alpha, const REAL beta, const REAL nu, REAL *myVarX, REAL *myVarY, REAL *myX, REAL *myY, int numX, int numY)
{
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidI >= numX || gidJ >= numY)
    return;

  myVarX[gidI * numY + gidJ] = 0.25 * exp(2.0 * (beta  * log(myX[gidI]) + myY[gidJ] + nu));
  myVarY[gidI * numY + gidJ] = 0.25 * exp(2.0 * (alpha * log(myX[gidI]) + myY[gidJ] + nu));

}

static __global__ void
set_payoff_kernel(REAL* myX, REAL* myResult, unsigned int numX, unsigned int numY, unsigned int numO)
{
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidI >= numX || gidJ >= numY || gidO >= numO)
    return;

  REAL strike = 0.001*gidO;
  REAL payoff = MAX(myX[gidI] - strike, (REAL)0.0);
  myResult[(gidO * numY + gidJ) * numX + gidI] = payoff;

}

static __global__ void
rollback_kernel_0(REAL *a, REAL *b, REAL *c, REAL *u, REAL *v, REAL *myResult, REAL *myVarX, REAL *myVarY, REAL *myDxx, REAL *myDyy, REAL dtInv, int numX, int numY)
{
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  const unsigned int bidI = blockIdx.x*blockDim.x;
  const unsigned int bidJ = blockIdx.y*blockDim.y;
  const unsigned int lidI = threadIdx.y;
  const unsigned int lidJ = threadIdx.x;
  const unsigned int gidI = bidI + lidI;
  const unsigned int gidJ = bidJ + lidJ;

  extern __shared__ char sh_mem1[];
  REAL *sh_mem   = (REAL*) sh_mem1;
  REAL *localDxx = (REAL*) sh_mem + 34*34;
  REAL *localDyy = (REAL*) localDxx + 32*3;

  if(bidI + lidJ >= numX || bidJ + lidI >= numY) {
    sh_mem[34*(lidJ+1) + lidI + 1] = 0.0;
  } else {
    sh_mem[34*(lidJ+1) + lidI + 1] = myResult[(gidO * numY + bidJ + lidI) * numX + bidI + lidJ];
  }

  if(lidI < 3) {
    if(gidJ < numY) {
      localDyy[lidI * 32 + lidJ] = myDyy[lidI * numY + gidJ];
    }
  } else if(lidI < 6) {
    if(bidI + lidJ < numX) {
      localDxx[(lidI - 3) * 32 + lidJ] = myDxx[(lidI - 3) * numX + bidI + lidJ];
    }
  } else if (lidI == 6) {
    if(bidJ > 0 && bidI + lidJ < numX) {
      sh_mem[34*(lidJ + 1)] = myResult[(gidO * numY + bidJ - 1) * numX + bidI + lidJ];
    } else {
      sh_mem[34*(lidJ + 1)] = 0.0;
    }
  } else if (lidI == 7) {
    if(bidJ + 32 < numY && bidI + lidJ < numX) {
      sh_mem[34*(lidJ + 1) + 33] = myResult[(gidO * numY + bidJ + 32) * numX + bidI + lidJ];
    } else {
      sh_mem[34*(lidJ + 1) + 33] = 0.0;
    }
  } else if (lidI == 8) {
    if(bidJ + lidJ < numY && bidI > 0) {
      sh_mem[lidJ + 1] = myResult[(gidO * numY + bidJ + lidJ) * numX + bidI - 1];
    } else {
      sh_mem[lidJ + 1] = 0.0;
    }
  } else if (lidI == 9) {
    if(bidJ + lidJ < numY && bidI + 32 < numX) {
      sh_mem[34*33 + lidJ + 1] = myResult[(gidO * numY + bidJ + lidJ) * numX + bidI + 32];
    } else {
      sh_mem[34*33 + lidJ + 1] = 0.0;
    }
  }

  __syncthreads();

  if(gidI >= numX || gidJ >= numY) {
    return;
  }

  v[(gidO * numX + gidI) * numY + gidJ] = 2.0 * myVarY[gidI * numY + gidJ] *
    (localDyy[0 * 32 + lidJ] * sh_mem[34*(lidI + 1) + lidJ] +
     localDyy[1 * 32 + lidJ] * sh_mem[34*(lidI + 1) + lidJ + 1] +
     localDyy[2 * 32 + lidJ] * sh_mem[34*(lidI + 1) + lidJ + 2]);

  u[(gidO * numX + gidI) * numY + gidJ] = myVarX[gidI * numY + gidJ] *
    (localDxx[0 * 32 + lidI] * sh_mem[34*lidI + lidJ + 1] +
     localDxx[1 * 32 + lidI] * sh_mem[34*(lidI + 1) + lidJ + 1] +
     localDxx[2 * 32 + lidI] * sh_mem[34*(lidI + 2) + lidJ + 1]) +
    v[(gidO * numX + gidI) * numY + gidJ] +
    dtInv * sh_mem[34*(lidI + 1) + lidJ + 1];

  if(gidO == 0) {
    a[gidI * numY + gidJ]  =       - myVarX[gidI * numY + gidJ] * localDxx[0 * 32 + lidI];
    b[gidI * numY + gidJ]  = dtInv - myVarX[gidI * numY + gidJ] * localDxx[1 * 32 + lidI];
    c[gidI * numY + gidJ]  =       - myVarX[gidI * numY + gidJ] * localDxx[2 * 32 + lidI];
  }
}

static __global__ void
rollback_kernel_1(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x + 1;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidJ >= numY)
    return;

  yy[gidI * numY + gidJ] = -a[gidI * numY + gidJ] * c[(gidI-1) * numY + gidJ];
}

static __global__ void
rollback_kernel_2(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  int i;

  if(gidJ >= numY)
    return;

  yy[gidJ] = 1.0 / b[gidJ];

  for(i = 1; i < numX; i++) {
    yy[i * numY + gidJ] = 1.0 / (b[i * numY + gidJ] + yy[i * numY + gidJ] * yy[(i-1) * numY + gidJ]);
  }
}


static __global__ void
rollback_kernel_3(REAL *a, REAL *b, REAL *c, REAL *u, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidJ >= numY)
    return;

  if(gidO == 0) {
    if(gidI > 0) {
      a[gidI * numY + gidJ] = 1.0 /
        (c[(gidI - 1) * numY + gidJ] *
         yy[(gidI-1) * numY + gidJ] -
         b[gidI * numY + gidJ] /
         a[gidI * numY + gidJ]
         );
    }
    b[gidI * numY + gidJ] = - c[gidI * numY + gidJ] * yy[gidI * numY + gidJ];
  }

  u[(gidO * numX + gidI) * numY + gidJ] = u[(gidO * numX + gidI) * numY + gidJ] * yy[gidI * numY + gidJ];
}

static __global__ void
rollback_kernel_4(REAL *u, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int i;

  if(gidJ >= numY)
    return;

  for(i = 1; i < numX; i++) {
    u[(gidO * numX + i) * numY + gidJ] += a[i * numY + gidJ] * u[(gidO * numX + i - 1) * numY + gidJ];
  }
  for(i = numX-2; i >= 0; i--) {
    u[(gidO * numX + i) * numY + gidJ] += b[i * numY + gidJ] * u[(gidO * numX + i + 1) * numY + gidJ];
  }
}

static __global__ void
rollback_kernel_5(REAL *a, REAL *b, REAL *c, REAL *y, REAL *u, REAL *v, REAL *myDyy, REAL *myVarY, REAL dtInv, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;

  if(gidJ >= numY)
    return;

  if(gidO == 0) {
    a[gidJ * numX + gidI] =       - myVarY[gidI * numY + gidJ] * myDyy[0 * numY + gidJ];
    b[gidJ * numX + gidI] = dtInv - myVarY[gidI * numY + gidJ] * myDyy[1 * numY + gidJ];
    c[gidJ * numX + gidI] =       - myVarY[gidI * numY + gidJ] * myDyy[2 * numY + gidJ];
  }
  y[(gidO * numX + gidI) * numY + gidJ] = dtInv * u[(gidO * numX + gidI) * numY + gidJ] - 0.5 * v[(gidO * numX + gidI) * numY + gidJ];

}

static __global__ void
rollback_kernel_6(REAL *a, REAL *c, REAL *yy, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidJ = blockIdx.y*blockDim.y + threadIdx.y;

  if(gidJ >= numY)
    return;

  if(gidJ > 0) {
    yy[gidJ * numX + gidI] = -a[gidJ * numX + gidI] * c[(gidJ - 1) * numX + gidI];
  }
}

static __global__ void
rollback_kernel_7(REAL *yy, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  int j;

  if(gidI >= numX)
    return;

  yy[gidI] = 1.0 / b[gidI];

  for(j = 1; j < numY; j++) {
    yy[j * numX + gidI] = 1.0 / (b[j * numX + gidI] + yy[j * numX + gidI] * yy[(j - 1) * numX + gidI]);
  }
}

static __global__ void
rollback_kernel_8(REAL *a, REAL *b, REAL *c, REAL *u, REAL *v, REAL *y, REAL *yy, REAL *myResult, int numX, int numY, REAL dtInv) {
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  const unsigned int bidI = blockIdx.x*blockDim.x;
  const unsigned int bidJ = blockIdx.y*blockDim.y;
  const unsigned int lidI = threadIdx.x;
  const unsigned int lidJ = threadIdx.y;
  const unsigned int gidI = bidI + lidI;
  const unsigned int gidJ = bidJ + lidJ;

  if(gidI >= numX)
    return;

  if(gidO == 0) {
    if(gidJ > 0) {
      a[gidJ * numX + gidI] = 1.0 /
        (c[(gidJ - 1) * numX + gidI] *
         yy[(gidJ - 1) * numX + gidI] -
         b[gidJ * numX + gidI] /
         a[gidJ * numX + gidI]);
    }
    b[gidJ * numX + gidI] = -c[gidJ * numX + gidI] * yy[gidJ * numX + gidI];
  }

  myResult[(gidO * numY + gidJ) * numX + gidI] = y[(gidO * numX + gidI) * numY + gidJ] * yy[gidJ * numX + gidI];
}

static __global__ void
rollback_kernel_9(REAL *myResult, REAL *a, REAL *b, int numX, int numY) {
  const unsigned int gidI = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int gidO = blockIdx.z*blockDim.z + threadIdx.z;
  int j;

  if(gidI >= numX)
    return;

  for(j = 1; j < numY; j++) {
    myResult[(gidO * numY + j) * numX + gidI] += a[j * numX + gidI] * myResult[(gidO * numY + j - 1) * numX + gidI];
  }
  for(j = numY-2; j >= 0; j--) {
    myResult[(gidO * numY + j) * numX + gidI] += b[j * numX + gidI] * myResult[(gidO * numY + j + 1) * numX + gidI];
  }
}

struct timeval t_start;
long long counters[20];

static void
start()
{
  gettimeofday(&t_start, NULL);
}

static void
end(long long *out)
{

  struct timeval t_end, t_diff;
  gettimeofday(&t_end, NULL);
  timersub(&t_end, &t_start, &t_diff);
  *out += t_diff.tv_sec*1e6+t_diff.tv_usec;
}

static void
rollback(const REAL dtInv, PrivGlobs &globs)
{

  /* v[o][i][j] = myDyy[0..2][j] `dot` myResult[o][j-1..j+1][i] * myVarY[i][j] * 2.0
     u[o][i][j] = myDxx[0..2][i] `dot` myResult[o][j][i-1..i+1] * myVarX[i][j] + v[o][i][j] + dtInv * myResult[o][j][i]
     a[i][j]    =       - myVarX[i][j] * myDxx[0][i]
     b[i][j]    = dtInv - myVarX[i][j] * myDxx[1][i]
     c[i][j]    =       - myVarX[i][j] * myDxx[2][i]
   */
  rollback_kernel_0
    <<<
    dim3(DIVUP(globs.numX, 32), DIVUP(globs.numY, 32), globs.numO),
    dim3(32, 32, 1),
    (34*34 + 32*6)*sizeof(REAL)
    >>>
    (globs.a, globs.b, globs.c, globs.u, globs.v, globs.myResult, globs.myVarX, globs.myVarY, globs.myDxx, globs.myDyy, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[0]); start();

  /* yy[i][j] = - a[i][j] * c[i-1][j] */
  rollback_kernel_1
    <<<
    dim3(globs.numX-1, DIVUP(globs.numY, 128), 1),
    dim3(1, 128, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[1]); start();

  /* yy[0][j] = 1.0 / b[0][j]
     yy[i][j] = 1.0 / (b[i][j] + yy[i][j] * yy[i-1][j])*/
  rollback_kernel_2
    <<<
    dim3(1, DIVUP(globs.numY, 128), 1),
    dim3(1, 128, 1)
    >>>
    (globs.yy, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[2]); start();

  /* a[i][j]    = 1.0 / (c[i-1][j] * yy[i-1][j] - b[i][j] / a[i][j])
     b[i][j]    = -  c[i][j] * yy[i][j]
     u[o][i][j] = u[o][i][j] * yy[i][j] */
  rollback_kernel_3
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 128), globs.numO),
    dim3(1, 128, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.u, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[3]); start();

  /* u[o][i][j] += a[i][j] * u[o][i-1][j]
     u[o][i][j] += b[i][j] * u[o][i+1][j] */
  rollback_kernel_4
    <<<
    dim3(1, DIVUP(globs.numY, 128), globs.numO),
    dim3(1, 128, 1)
    >>>
    (globs.u, globs.a, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[4]); start();

  /* a[j][i] =  =       - 0.25 * myVarY[j][i] * myDyy[0][j]
     b[j][i] =  = dtInv - 0.25 * myVarY[j][i] * myDyy[1][j]
     c[j][i] =  =       - 0.25 * myVarY[j][i] * myDyy[2][j]
     y[o][i][j] = dtInv * u[o][i][j] - 0.5 * v[o][i][j] */
  rollback_kernel_5
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 128), globs.numO),
    dim3(1, 128, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.u, globs.v, globs.y, globs.myDyy, globs.myVarY, dtInv, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[5]); start();

  /* yy[j][i] = -a[j][i] * c[j-1][i] */
  rollback_kernel_6
    <<<
    dim3(DIVUP(globs.numX, 128), globs.numY, 1),
    dim3(128, 1, 1)
    >>>
    (globs.a, globs.c, globs.yy, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[6]); start();

  /* yy[0][i] = 1.0 / b[0][i]
     yy[j][i] = 1.0 / (b[j][i] + yy[j][i] * yy[j-1][i]) */
  rollback_kernel_7
    <<<
    dim3(DIVUP(globs.numX, 128), 1, 1),
    dim3(128, 1, 1)
    >>>
    (globs.yy, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[7]); start();

  /* a[j][i] = 1.0 / (c[j-1][i] * yy[j-1][i] - b[j][i] / a[j][i])
     b[j][i] = -c[j][i] * yy[j][i]
     myResult[o][j][i] = y[o][i][j] * yy[j][i] */
  rollback_kernel_8
    <<<
    dim3(DIVUP(globs.numX, 128), globs.numY, globs.numO),
    dim3(128, 1, 1)
    >>>
    (globs.a, globs.b, globs.c, globs.u, globs.v, globs.y, globs.yy, globs.myResult, globs.numX, globs.numY, dtInv);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[8]); start();

  /* myResult[o][j][i] += a[j][i] * myResult[o][j-1][i]
     myResult[o][j][i] += b[j][i] * myResult[o][j+1][i] */
  rollback_kernel_9
    <<<
    dim3(DIVUP(globs.numX, 128), 1, globs.numO),
    dim3(128, 1, 1)
    >>>
    (globs.myResult, globs.a, globs.b, globs.numX, globs.numY);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaThreadSynchronize());
  end(&counters[9]);

}

static void
value(PrivGlobs &globs,
      const REAL s0,
      const REAL t,
      const REAL alpha,
      const REAL nu,
      const REAL beta,
      REAL *res)
{

  set_payoff_kernel
    <<<
    dim3(globs.numX, DIVUP(globs.numY, 128), globs.numO),
    dim3(1, 128, 1)
    >>>
    (globs.myX, globs.myResult, globs.numX, globs.numY, globs.numO);

  for(int i = globs.numT-2; i >= 0; i--) {
    start();

    updateParams_kernel
      <<<
      dim3(globs.numX, DIVUP(globs.numY, 128), 1),
      dim3(1, 128, 1)
      >>>
      (alpha, beta, -0.5 * nu * nu * globs.myTimeline[i], globs.myVarX, globs.myVarY, globs.myX, globs.myY, globs.numX, globs.numY);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaThreadSynchronize());
    end(&counters[10]); start();

    rollback(1.0 / (globs.myTimeline[i+1] - globs.myTimeline[i]), globs);
  }

  for(int i = 0; i < globs.numO; i++) {
    cudaMemcpy(&res[i], &globs.myResult[(i * globs.numY + globs.myYindex)*globs.numX + globs.myXindex], sizeof(REAL), cudaMemcpyDeviceToHost);
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
  initOperator(globs.myX, numX, globs.myDxx, outer);
  initOperator(globs.myY, numY, globs.myDyy, outer);

  for(int i = 0; i <= 10; i++) {
    counters[i] = 0;
  }

  value(globs, s0,   t,
        alpha, nu,   beta,
        res);
  for(int i = 0; i <= 10; i++) {
    printf("%lld %d\n", counters[i], i);
  }
}
