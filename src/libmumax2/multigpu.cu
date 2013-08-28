#include "multigpu.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

int _Ngpu = 0;
int* _gpuIds = NULL;

__export__ void setUsedGPUs(int* gpuIds, int Ngpu)
{
    assert(_Ngpu == 0);
    assert(_gpuIds == NULL);
    _Ngpu = Ngpu;
    _gpuIds = (int*)calloc(Ngpu, sizeof(int));
    for(int i = 0; i < Ngpu; i++)
    {
        _gpuIds[i] = gpuIds[i];
    }
}



int nDevice()
{
    assert(_Ngpu != 0);
    assert(_gpuIds != NULL);
    return _Ngpu;
}

int deviceId(int i)
{
    assert(i >= 0 && i < _Ngpu);
    assert(_Ngpu != 0);
    assert(_gpuIds != NULL);
    return _gpuIds[i];
}


#ifdef __cplusplus
}
#endif
