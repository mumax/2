#ifndef _MULTIGPU_H_
#define _MULTIGPU_H_

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the number of GPUs used for this simulation.
int nDevice();

/// Returns the device ID of the i-th GPU used.
/// E.g.: when GPUs 2 and 3 are used, deviceId(0) will return 2.
int deviceId(int i);

__declspec(dllexport) void setUsedGPUs(int* gpuIds, int Ngpu);

#ifdef __cplusplus
}
#endif
#endif
