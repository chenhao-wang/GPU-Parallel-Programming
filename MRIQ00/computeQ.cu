#include <cstdlib>

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 1024

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 512
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024



struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

__global__ void
ComputePhiMag_GPU(float* phiR, float* phiI, float* phiMag, int numK) {
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}
__global__ void ComputeQ_GPU(int numK, int kGlobalIndex, float* x, float* y, float* z, float* Qr, float* Qi, kValues* ck)
{
  
  int xIndex = blockIdx.x*KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;
  
  
  int kIndex = 0;
  for (kIndex=0; kIndex < KERNEL_Q_K_ELEMS_PER_GRID; kIndex++)
  {  float expArg = PIx2 * (ck[kIndex].Kx * x[xIndex] +
			   ck[kIndex].Ky * y[xIndex] +
			   ck[kIndex].Kz * z[xIndex]);
    Qr[xIndex] += ck[kIndex].PhiMag * cos(expArg);
    Qi[xIndex] += ck[kIndex].PhiMag * sin(expArg);
  }

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
  int QGrids = (numK-1) / KERNEL_Q_K_ELEMS_PER_GRID+1;
  int QBlocks = (numX-1) / KERNEL_Q_THREADS_PER_BLOCK+1;
  dim3 DimQBlock(KERNEL_Q_THREADS_PER_BLOCK, 1);
  dim3 DimQGrid(QBlocks, 1);
  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    ComputeQ_GPU <<< DimQGrid, DimQBlock >>>
		(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d, kValsTile);
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) malloc(numK * sizeof(float));
  *Qr = (float*) malloc(numX * sizeof (float));
  *Qi = (float*) malloc(numX * sizeof (float));
}


