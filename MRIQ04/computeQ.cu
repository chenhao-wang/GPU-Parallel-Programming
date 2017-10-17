#include <cstdlib>

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 512
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024


struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

__constant__ __device__ kValues ck[KERNEL_Q_K_ELEMS_PER_GRID];

__global__ void
ComputePhiMag_GPU(float* phiR, float* phiI, float* phiMag, int numK) {
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

__global__ void
ComputeQ_GPU(int numK, int kGlobalIndex,
	     float* x, float* y, float* z, float* Qr , float* Qi)
{
  __shared__ float sx,sy,sz,sQr,sQi;

  int xIndex = blockIdx.x*KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;

  sx = x[xIndex];
  sy = y[xIndex];
  sz = z[xIndex];
  sQr = Qr[xIndex];
  sQi = Qi[xIndex];

  int kIndex = 0;
  if (numK % 4)
 {
    for (int j=0;j<numK%4;j++)
   {
    float expArg = PIx2 * (ck[j].Kx * sx + ck[j].Ky * sy + ck[j].Kz * sz);
    sQr += ck[j].PhiMag * cos(expArg);
    sQi += ck[j].PhiMag * sin(expArg);
    kIndex++;
    kGlobalIndex++;
   }
 }

  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex += 4, kGlobalIndex += 4) {
    float expArg = PIx2 * (ck[kIndex].Kx * sx +
			   ck[kIndex].Ky * sy +
			   ck[kIndex].Kz * sz);
    sQr += ck[kIndex].PhiMag * cos(expArg);
    sQi += ck[kIndex].PhiMag * sin(expArg);

    int kIndex1 = kIndex + 1;
    float expArg1 = PIx2 * (ck[kIndex1].Kx * sx +
			    ck[kIndex1].Ky * sy +
			    ck[kIndex1].Kz * sz);
    sQr += ck[kIndex1].PhiMag * cos(expArg1);
    sQi += ck[kIndex1].PhiMag * sin(expArg1);

    int kIndex2 =kIndex+ 2;
    float expArg2 = PIx2 * (ck[kIndex2].Kx * sx +
			    ck[kIndex2].Ky * sy +
			    ck[kIndex2].Kz * sz);
    sQr += ck[kIndex2].PhiMag * cos(expArg2);
    sQi += ck[kIndex2].PhiMag * sin(expArg2);   

    int kIndex3 =kIndex+ 3;
    float expArg3 = PIx2 * (ck[kIndex3].Kx * sx +
			    ck[kIndex3].Ky * sy +
			    ck[kIndex3].Kz * sz);
    sQr += ck[kIndex3].PhiMag * cos(expArg3);
    sQi += ck[kIndex3].PhiMag * sin(expArg3);   
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
}

void computePhiMag_GPU(int numK, float* phiR_d, float* phiI_d, float* phiMag_d)
{
  int phiMagBlocks = (numK-1) / KERNEL_PHI_MAG_THREADS_PER_BLOCK+1;
  dim3 DimPhiMagBlock(KERNEL_PHI_MAG_THREADS_PER_BLOCK, 1);
  dim3 DimPhiMagGrid(phiMagBlocks, 1);

  ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> 
    (phiR_d, phiI_d, phiMag_d, numK);
}

void computeQ_GPU(int numK, int numX,
                  float* x_d, float* y_d, float* z_d,
                  kValues* kVals,
                  float* Qr_d, float* Qi_d)
{
  int QGrids = (numK-1)/KERNEL_Q_K_ELEMS_PER_GRID+1;
  int QBlocks =(numX-1)/KERNEL_Q_THREADS_PER_BLOCK+1;
  dim3 DimQBlock(KERNEL_Q_THREADS_PER_BLOCK, 1);
  dim3 DimQGrid(QBlocks, 1);

  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);
    cudaMemcpyToSymbol(ck, kValsTile, numElems * sizeof(kValues), 0);
    ComputeQ_GPU <<< DimQGrid, DimQBlock >>>
      (numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) malloc(numK * sizeof(float));
  *Qr = (float*) malloc(numX * sizeof (float));
  *Qi = (float*) malloc(numX * sizeof (float));
}
