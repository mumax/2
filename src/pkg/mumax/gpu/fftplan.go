//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste and Ben Van de Wiele

import (
	. "mumax/common"
	cu "cuda/driver"
	"cuda/cufft"
	"fmt"
)

type FFTPlan struct {
	dataSize  [3]int         // Size of the (non-zero) input data block
	logicSize [3]int         // Transform size including zero-padding. >= dataSize
	padZ      Array          // Buffer for Z-zeropadding and +2 elements for R2C
	planZ     []cufft.Handle // In-place transform of padZ parts, 1/GPU /// ... from outer space
	transp1   Array          // Buffer for partial transpose per GPU
	chunks    []Array        // A chunk (single-GPU part of these arrays) is copied from GPU to GPU
	transp2   Array          // Buffer for full YZ inter device transpose + zero padding in Z' and X
	planYX    []cufft.Handle // In-place transform of transp2 parts. Is just a Y transform for 2D.
	planY     []cufft.Handle // In-place transform of transp2 parts, in y-direction
	planX     []cufft.Handle // In-place transform of transp2 parts, in x-direction (strided)
	Stream                   //
}

func (fft *FFTPlan) Init(dataSize, logicSize []int) {
	Assert(len(dataSize) == 3)
	Assert(len(logicSize) == 3)
	NDev := NDevice()
	const nComp = 1

	// init size
	for i := range fft.dataSize {
		fft.dataSize[i] = dataSize[i]
		fft.logicSize[i] = logicSize[i]
	}

	// init stream
	fft.Stream = NewStream()

	// init padZ
	padZN0 := fft.dataSize[0]
	padZN1 := fft.dataSize[1]
	padZN2 := fft.logicSize[2] + 2
	fft.padZ.Init(nComp, []int{padZN0, padZN1, padZN2}, DO_ALLOC)

	// init planZ
	fft.planZ = make([]cufft.Handle, NDev)
	for dev := range _useDevice {
		setDevice(_useDevice[dev])
		Assert((nComp*padZN0*padZN1)%NDev == 0)
		fft.planZ[dev] = cufft.Plan1d(fft.logicSize[2], cufft.R2C, (nComp*padZN0*padZN1)/NDev)
		fft.planZ[dev].SetStream(uintptr(fft.Stream[dev])) // TODO: change
	}

	// init transp1
	fft.transp1.Init(nComp, fft.padZ.size3D, DO_ALLOC)

	// init chunks
	chunkN0 := dataSize[0]
	Assert((logicSize[2]/2)%NDev == 0)
	chunkN1 := ((logicSize[2]/2)/NDev + 1) * NDev // (complex numbers)
	Assert(dataSize[1]%NDev == 0)
	chunkN2 := (dataSize[1] / NDev) * 2 // (complex numbers)
	fft.chunks = make([]Array, NDev)
	for dev := range _useDevice {
		fft.chunks[dev].Init(nComp, []int{chunkN0, chunkN1, chunkN2}, DO_ALLOC)
	}

	// init transp2
	transp2N0 := dataSize[0] // make this logicSize[0] when copyblock can handle it
	Assert((logicSize[2]+2*NDev)%2 == 0)
	transp2N1 := (logicSize[2] + 2*NDev) / 2
	transp2N2 := logicSize[1] * 2
	fft.transp2.Init(nComp, []int{transp2N0, transp2N1, transp2N2}, DO_ALLOC)

	// init planYX
	fft.planYX = make([]cufft.Handle, NDev)
	for dev := range _useDevice {
		setDevice(_useDevice[dev])
		if fft.logicSize[0] == 1 { // 2D
//			batch := (fft.transp2.size3D[0] * fft.transp2.size3D[1]) / NDev
      batchY := (fft.logicSize[2])/2/NDev + 1
			//fmt.Println("batch=", batch)
			fft.planYX[dev] = cufft.PlanMany([]int{fft.logicSize[1]}, nil, 1, nil, 1, cufft.C2C, batchY)
		} else { //3D
//			panic("3D?")
      batchY := ( (fft.logicSize[2])/2/NDev + 1 ) * fft.logicSize[0]
			fft.planY[dev] = cufft.PlanMany([]int{fft.logicSize[1]}, nil, 1, nil, 1, cufft.C2C, batchY)
			batchX := ( (fft.logicSize[2])/2/NDev + 1 ) * fft.logicSize[1];
      fft.planX[dev] = cufft.PlanMany([]int{fft.logicSize[0]}, nil, fft.logicSize[0], nil, fft.logicSize[0], cufft.C2C, batchX)
		}
		fft.planYX[dev].SetStream(uintptr(fft.Stream[dev])) // TODO: change 
	}

}

func NewFFTPlan(dataSize, logicSize []int) *FFTPlan {
	fft := new(FFTPlan)
	fft.Init(dataSize, logicSize)
	return fft
}

func (fft *FFTPlan) Free() {
	for i := range fft.dataSize {
		fft.dataSize[i] = 0
		fft.logicSize[i] = 0
	}
	(&(fft.padZ)).Free()

	// TODO destroy
}

func (fft *FFTPlan) Normalization() int {
	return (fft.logicSize[X] * fft.logicSize[Y] * fft.logicSize[Z])
}

func (fft *FFTPlan) Forward(in, out *Array) {
	AssertMsg(in.size4D[0] == 1, "1")
	AssertMsg(out.size4D[0] == 1, "2")
	AssertMsg(in.size3D[0] == fft.dataSize[0], "3")
	AssertMsg(in.size3D[1] == fft.dataSize[1], "4")
	AssertMsg(in.size3D[2] == fft.dataSize[2], "5")
	AssertMsg(out.size3D[0] == fft.logicSize[0], "6")
	AssertMsg(out.size3D[1] == fft.logicSize[1], "7")
	AssertMsg(out.size3D[2] == fft.logicSize[2]+2, "8")

	// shorthand
	padZ := &(fft.padZ)
	transp1 := &(fft.transp1)
	dataSize := fft.dataSize
	logicSize := fft.logicSize
	NDev := NDevice()
	chunks := fft.chunks // not sure if chunks[0] copies the struct...
	transp2 := &(fft.transp2)

	fmt.Println("in:", in.LocalCopy().Array)

	Start("CopyPadZ")
	CopyPadZ(padZ, in)
	Stop("CopyPadZ")

	fmt.Println("padZ:", padZ.LocalCopy().Array)

	// fft Z
	Start("fftZ")
	for dev := range _useDevice {
		fft.planZ[dev].ExecR2C(uintptr(padZ.pointer[dev]), uintptr(padZ.pointer[dev])) // is this really async?
	}
	fft.Sync()
	Stop("fftZ")
	fmt.Println("fftZ:", padZ.LocalCopy().Array)

	Start("Transpose1")
	TransposeComplexYZPart(transp1, padZ) // fftZ!
	Stop("Transpose1")
	fmt.Println("transp1:", transp1.LocalCopy().Array)

	// copy chunks, cross-device
	Start("MemcpyDtoD")
	chunkPlaneBytes := int64(chunks[0].partSize[1]*chunks[0].partSize[2]) * SIZEOF_FLOAT // one plane 
	Assert(dataSize[1]%NDev == 0)
	Assert(logicSize[2]%NDev == 0)
	for dev := range _useDevice { // source device
		for c := range chunks { // source chunk
			// source device = dev
			// target device = chunk

			for i := 0; i < dataSize[0]; i++ { // only memcpys in this loop
				srcPlaneN := transp1.partSize[1] * transp1.partSize[2] //fmt.Println("srcPlaneN:", srcPlaneN)//seems OK
				srcOffset := i*srcPlaneN + c*((dataSize[1]/NDev)*(logicSize[2]/NDev))
				src := cu.DevicePtr(ArrayOffset(uintptr(transp1.pointer[dev]), srcOffset))

				dstPlaneN := chunks[0].partSize[1] * chunks[0].partSize[2] //fmt.Println("dstPlaneN:", dstPlaneN)//seems OK
				dstOffset := i * dstPlaneN
				dst := cu.DevicePtr(ArrayOffset(uintptr(chunks[dev].pointer[c]), dstOffset))

				cu.MemcpyDtoD(dst, src, chunkPlaneBytes) // chunkPlaneBytes for plane-by-plane
			}
		}
	}
	Stop("MemcpyDtoD")

	Start("InsertBlockZ")
	transp2.Zero()
	for c := range chunks {
		InsertBlockZ(transp2, &(chunks[c]), c) // no need to offset planes here.
	}
	Stop("InsertBlockZ")
	fmt.Println("transp2:", transp2.LocalCopy().Array)

	// FFT Y(X)
	Start("fftYX")
	if logicSize[0] == 1 { //2D
    for dev := range _useDevice {
      fft.planYX[dev].ExecC2C(uintptr(transp2.pointer[dev]), uintptr(out.pointer[dev]), cufft.FORWARD) // is this really async?
    }
  } else{ //3D
    for dev := range _useDevice {
      fft.planY[dev].ExecC2C(uintptr(transp2.pointer[dev]), uintptr(out.pointer[dev]), cufft.FORWARD) //FFT in y-direction
    }
    fft.Sync()   
    for dev := range _useDevice {
      fft.planX[dev].ExecC2C(uintptr(transp2.pointer[dev]), uintptr(out.pointer[dev]), cufft.FORWARD) //FFT in x-direction
    }
  }
	fft.Sync()
	Stop("fftYX")
	fmt.Println("out:", out.LocalCopy().Array)
}

// DOES NOT WORK YET
func (fft *FFTPlan) Inverse(in, out *Array) {
	// shorthand
	padZ := &(fft.padZ)
	transp1 := &(fft.transp1)
	dataSize := fft.dataSize
	logicSize := fft.logicSize
	NDev := NDevice()
	chunks := fft.chunks // not sure if chunks[0] copies the struct...
	transp2 := &(fft.transp2)

	fmt.Println("transp2:", transp2.LocalCopy().Array)
	// FFT Y(X)
	for dev := range _useDevice {
		fft.planYX[dev].ExecC2C(uintptr(transp2.pointer[dev]), uintptr(transp2.pointer[dev]), cufft.INVERSE) // is this really async?
	}
	fft.Sync()

	for c := range chunks {
		InsertBlockZ(transp2, &(chunks[c]), c) // UNBLOCK
	}

	fmt.Println("transp2:", transp2.LocalCopy().Array)

	// copy chunks, cross-device
	chunkPlaneBytes := int64(chunks[0].partSize[1]*chunks[0].partSize[2]) * SIZEOF_FLOAT // one plane 
	for dev := range _useDevice {                                                        // source device
		for c := range chunks { // source chunk
			// source device = dev
			// target device = chunk

			for i := 0; i < dataSize[0]; i++ { // only memcpys in this loop
				srcPlaneN := transp1.partSize[1] * transp1.partSize[2] //fmt.Println("srcPlaneN:", srcPlaneN)//seems OK
				srcOffset := i*srcPlaneN + c*((dataSize[1]/NDev)*(logicSize[2]/NDev))
				src := cu.DevicePtr(ArrayOffset(uintptr(transp1.pointer[dev]), srcOffset))

				dstPlaneN := chunks[0].partSize[1] * chunks[0].partSize[2] //fmt.Println("dstPlaneN:", dstPlaneN)//seems OK
				dstOffset := i * dstPlaneN
				dst := cu.DevicePtr(ArrayOffset(uintptr(chunks[dev].pointer[c]), dstOffset))
				// must be done plane by plane
				cu.MemcpyDtoD(dst, src, chunkPlaneBytes) // chunkPlaneBytes for plane-by-plane
			}
		}
	}

	TransposeComplexYZPart(transp1, padZ) // fftZ!
	//(&transp1).CopyFromDevice(&padZ)
	fmt.Println("transp1:", transp1.LocalCopy().Array)

	// fft Z
	for dev := range _useDevice {
		fft.planZ[dev].ExecR2C(uintptr(padZ.pointer[dev]), uintptr(padZ.pointer[dev])) // is this really async?
	}
	fft.Sync()
	fmt.Println("fftZ:", padZ.LocalCopy().Array)

	CopyPadZ(padZ, in)
	fmt.Println("padZ:", padZ.LocalCopy().Array)

}
