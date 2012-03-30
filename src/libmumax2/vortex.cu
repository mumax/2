#include "uniform.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void initVectorQuantVortexRegionKern(float* Sx, float* Sy, float* Sz,
												float* regions,
												bool* regionsToProceed,
												float centerX, float centerY, float centerZ,
												float axisX, float axisY, float axisZ,
												float cellSizeX, float cellSizeY, float cellSizeZ,
												int polarity,
												int chirality,
												float maxRadius,
												int regionNum,
												int Npart,
												int dev,
												int Nx, int NyPart, int Nz) {
	int i = threadindex;
	if (i < Npart) {
		int regionIndex = __float2int_rn(regions[i]);

		/*Sx[threadindex] = ( __int2float_rn(i%Nx) + 0.5 ) * cellSizeX;
		Sy[threadindex] = ( __int2float_rn(((i/Nx)%NyPart+(NyPart*dev))) + 0.5 ) * cellSizeY;
		Sz[threadindex] = ( __int2float_rn(i/(Nx*NyPart)) + 0.5 ) * cellSizeZ;*/
		if (regionIndex < regionNum && regionIndex > 0 && regionsToProceed[regionIndex]==true) {
			// component of v the shortest vector going from the line (passing by center and direction axis) and the current point
			float v1X = centerX - ( __int2float_rn(i%Nx) + 0.5 ) * cellSizeX;
			float v1Y = centerY - ( __int2float_rn(((i/Nx)%NyPart+(NyPart*dev))) + 0.5 ) * cellSizeY;
			float v1Z = centerZ - ( __int2float_rn(i/(Nx*NyPart)) + 0.5 ) * cellSizeZ;
			// scalar product v.u
			float vScalaru = v1X * axisX + v1Y * axisY + v1Z * axisZ;
			v1X -= axisX * vScalaru;
			v1Y -= axisY * vScalaru;
			v1Z -= axisZ * vScalaru;
			// v norm == distance between voxel and vortex axis
			float d = sqrt(v1X * v1X + v1Y * v1Y + v1Z * v1Z);
			if (maxRadius == 0. || d <= maxRadius) {
				// set field to vortex
				if (d < 5e-8) {
					float gauss = __expf(-1.0 * d * d / 25.0e-18);
					Sx[threadindex] = axisX * polarity * gauss  - (1-gauss) * chirality * ( axisY * v1Z - axisZ * v1Y)/d;
					Sy[threadindex] = axisY * polarity * gauss  - (1-gauss) * chirality * ( axisZ * v1X - axisX * v1Z)/d;
					Sz[threadindex] = axisZ * polarity * gauss  - (1-gauss) * chirality * ( axisX * v1Y - axisY * v1X)/d;
				}
				else{
					Sx[threadindex] = - chirality * ( axisY * v1Z - axisZ * v1Y)/d;
					Sy[threadindex] = - chirality * ( axisZ * v1X - axisX * v1Z)/d;
					Sz[threadindex] = - chirality * ( axisX * v1Y - axisY * v1X)/d;
				}
			}
		}
	}
}

__export__ void initVectorQuantVortexRegionAsync(float** Sx, float** Sy, float** Sz,
									  float** regions,
									  bool* host_regionsToProceed,
									  float centerX, float centerY, float centerZ,
									  float axisX, float axisY, float axisZ,
									  float cellsizeX, float cellsizeY, float cellsizeZ,
									  int polarity,
									  int chirality,
									  float maxRadius,
									  int initValNum,
									  CUstream* stream,
									  int Npart,
									  int Nx, int NyPart, int Nz) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	bool* dev_regionsToProceed;
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(Sx[dev] != NULL);
		assert(Sy[dev] != NULL);
		assert(Sz[dev] != NULL);
		assert(regions[dev] != NULL);
		assert(host_regionsToProceed != NULL);
		gpu_safe(cudaSetDevice(deviceId(dev)));
		gpu_safe( cudaMalloc( (void**)&dev_regionsToProceed,initValNum * sizeof(bool)));
		gpu_safe( cudaMemcpy(dev_regionsToProceed,host_regionsToProceed,initValNum * sizeof(bool), cudaMemcpyHostToDevice));
		initVectorQuantVortexRegionKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Sx[dev],Sy[dev],Sz[dev],
																								 regions[dev],
																								 dev_regionsToProceed,
																								 centerX,centerY,centerZ,
																								 axisX,axisY,axisZ,
																								 cellsizeX,cellsizeY,cellsizeZ,
																								 polarity,
																								 chirality,
																								 maxRadius,
																								 initValNum,
																								 Npart,
																								 dev,
																								 Nx, NyPart,Nz);
		cudaFree(dev_regionsToProceed);
	}
}

#ifdef __cplusplus
}
#endif
