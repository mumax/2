//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Authors: Arne Vansteenkiste and Ben Van de Wiele

import (
	. "mumax/common"
	cu "cuda/driver"
	"cuda/cufft"
		"fmt"
	//   "cuda/runtime"
)

//Register this FFT plan
func init() {
	fftPlans["5"] = NewFFTPlan5
}

// runtime.GetDeviceProperties().MultiProcessorCount
type FFTPlan5 struct {
	//sizes
	dataSize   [3]int // Size of the (non-zero) input data block
	logicSize  [3]int // Transform size including zero-padding. >= dataSize
	outputSize [3]int // Size of the output data (one extra row PER GPU)

	//buffer, allocated
	buffer Array // Buffer for Z-zeropadding and +2 elements for R2C

	//arrays, not allocated
	padZ       Array   // Array for Z-zeropadding and +2 elements for R2C
	fftZbuffer Array   // Array after R2C transform: same size as padZ +2 elements in z-direction
	transp1    Array   // Array for partial transpose per GPU
	chunks     []Array // A chunk (single-GPU part of these arrays) is copied from GPU to GPU
	transp2    Array   // Arrays for full YZ inter device transpose + zero padding in Z' and X
	fftZ1Dev   []Array // Arrays containing data for batched FFTs when 1 device is used.

	// fft plans
	planY     []cufft.Handle // In-place transform of transp2 parts, in y-direction
	planX     []cufft.Handle // In-place transform of transp2 parts, in x-direction (strided)
	planZ_FW  []cufft.Handle // Forward transform of padZ parts, 1/GPU /// ... from outer space
	planZ_INV []cufft.Handle // Inverse transform of padZ parts, 1/GPU /// ... from outer space
	Stream                   //
}

func (fft *FFTPlan5) init(dataSize, logicSize []int) {
	Assert(len(dataSize) == 3)
	Assert(len(logicSize) == 3)
	NDev := NDevice()
	const nComp = 1

	// init size ------------------------------------
	outputSize := FFTOutputSize(logicSize)
	for i := range fft.dataSize {
		fft.dataSize[i] = dataSize[i]
		fft.logicSize[i] = logicSize[i]
		fft.outputSize[i] = outputSize[i]
	} //---------------------------------------------

	// init stream ----------------------------------
	fft.Stream = NewStream()
	//-----------------------------------------------

	if NDev == 1 { //  single-gpu implementation

		offset := ((fft.logicSize[2])/2 + 1) * fft.logicSize[1]
		fft.fftZ1Dev = make([]Array, fft.dataSize[0])
		for i := 0; i < fft.dataSize[0]; i++ {
			fft.fftZ1Dev[i].Init(nComp, []int{1, 1, offset}, DONT_ALLOC)
		} //---------------------------------------------

		// init planZ -----------------------------------
		fft.planZ_FW = make([]cufft.Handle, NDev)
		fft.planZ_INV = make([]cufft.Handle, NDev)
		fft.planZ_FW[0] = cufft.Plan1d(fft.logicSize[2], cufft.R2C, fft.dataSize[1])
		fft.planZ_FW[0].SetStream(uintptr(fft.Stream[0]))
		fft.planZ_INV[0] = cufft.Plan1d(fft.logicSize[2], cufft.C2R, fft.dataSize[1])
		fft.planZ_INV[0].SetStream(uintptr(fft.Stream[0]))
		//-----------------------------------------------

		// init planY -----------------------------------
		fft.planY = make([]cufft.Handle, NDev)
		batchY := ((fft.logicSize[2])/2 + 1)
		strideY := ((fft.logicSize[2])/2 + 1)
		fft.planY[0] = cufft.PlanMany([]int{fft.logicSize[1]}, []int{1}, strideY, []int{1}, strideY, cufft.C2C, batchY)
		fft.planY[0].SetStream(uintptr(fft.Stream[0]))

		if fft.logicSize[0] == 1 { // 2D
			fft.planX = nil
		} else { //3D
			fft.planX = make([]cufft.Handle, NDev)
			batchX := ((fft.logicSize[2])/2 + 1) * fft.logicSize[1]
			strideX := ((fft.logicSize[2])/2 + 1) * fft.logicSize[1]
			fft.planX[0] = cufft.PlanMany([]int{fft.logicSize[0]}, []int{1}, strideX, []int{1}, strideX, cufft.C2C, batchX)
			fft.planX[0].SetStream(uintptr(fft.Stream[0]))
		} //--------------------------------------------


	} else { // multi-gpu implementation

		//     fft.fftZ1Dev = nil  TODO  How to give this a null pointer?

		// init buffer (allocated) ----------------------
		bufferSize := dataSize[0] * ((logicSize[2]/2)/NDev + 1) * dataSize[1] * 2 //this size is the one needed for the chuncks
		fft.buffer.Init(nComp, []int{1, NDev, bufferSize}, DO_ALLOC)
		//-----------------------------------------------

		// init padZ (not allocated)  -------------------
		padZN0 := fft.dataSize[0]
		padZN1 := fft.dataSize[1]
		fft.padZ.Init(nComp, []int{fft.dataSize[0], fft.dataSize[1], fft.logicSize[2]}, DONT_ALLOC)
		fft.fftZbuffer.Init(nComp, []int{fft.dataSize[0], fft.dataSize[1], fft.logicSize[2] + 2}, DONT_ALLOC)
		//-----------------------------------------------

		// init transp1 (not allocated) -----------------
		fft.transp1.Init(nComp, fft.fftZbuffer.size3D, DONT_ALLOC)
		//-----------------------------------------------

		// init chunks (not allocated) ------------------
		chunkN0 := dataSize[0]
		Assert((logicSize[2]/2)%NDev == 0)
		chunkN1 := ((logicSize[2]/2)/NDev + 1) * NDev // (complex numbers)
		Assert(dataSize[1]%NDev == 0)
		chunkN2 := (dataSize[1] / NDev) * 2 // (complex numbers)
		fft.chunks = make([]Array, NDev)
		for dev := range _useDevice {
			fft.chunks[dev].Init(nComp, []int{chunkN0, chunkN1, chunkN2}, DONT_ALLOC)
		} //---------------------------------------------

		// init transp2 (not allocated) -----------------
		transp2N0 := dataSize[0] // make this logicSize[0] when copyblock can handle it
		Assert((logicSize[2]+2*NDev)%2 == 0)
		transp2N1 := (logicSize[2] + 2*NDev) / 2
		transp2N2 := logicSize[1] * 2
		fft.transp2.Init(nComp, []int{transp2N0, transp2N1, transp2N2}, DONT_ALLOC) //TODO make this point to the output array
		//-----------------------------------------------

		// init planZ -----------------------------------
		fft.planZ_FW = make([]cufft.Handle, NDev)
		fft.planZ_INV = make([]cufft.Handle, NDev)
		for dev := range _useDevice {
			setDevice(_useDevice[dev])
			Assert((nComp*padZN0*padZN1)%NDev == 0)
			fft.planZ_FW[dev] = cufft.Plan1d(fft.logicSize[2], cufft.R2C, (nComp*padZN0*padZN1)/NDev)
			fft.planZ_FW[dev].SetStream(uintptr(fft.Stream[dev])) // TODO: change
			fft.planZ_INV[dev] = cufft.Plan1d(fft.logicSize[2], cufft.C2R, (nComp*padZN0*padZN1)/NDev)
			fft.planZ_INV[dev].SetStream(uintptr(fft.Stream[dev])) // TODO: change
		} //--------------------------------------------

		// init planY -----------------------------------
		fft.planY = make([]cufft.Handle, NDev)
		batchY := ((fft.logicSize[2])/2/NDev + 1) * fft.dataSize[0]
		for dev := range _useDevice {
			setDevice(_useDevice[dev])
			fft.planY[dev] = cufft.PlanMany([]int{fft.logicSize[1]}, nil, 1, nil, 1, cufft.C2C, batchY)
			fft.planY[dev].SetStream(uintptr(fft.Stream[dev])) // TODO: change 
		} //--------------------------------------------

		// init planX -----------------------------------
		if fft.logicSize[0] == 1 { // 2D
			fft.planX = nil
		} else { //3D
			fft.planX = make([]cufft.Handle, NDev)
			batchX := ((fft.logicSize[2])/2/NDev + 1) * fft.logicSize[1]
			stride := batchX
			for dev := range _useDevice {
				setDevice(_useDevice[dev])
				fft.planX[dev] = cufft.PlanMany([]int{fft.logicSize[0]}, []int{1}, stride, []int{1}, stride, cufft.C2C, batchX)
				fft.planX[dev].SetStream(uintptr(fft.Stream[dev])) // TODO: change
			}
		} //--------------------------------------------
	}

}

func NewFFTPlan5(dataSize, logicSize []int) FFTInterface {
	fft := new(FFTPlan5)
	fft.init(dataSize, logicSize)
	return fft
}

func (fft *FFTPlan5) Free() {
	for i := range fft.dataSize {
		fft.dataSize[i] = 0
		fft.logicSize[i] = 0
	}
	if NDevice() > 1 {
		(&(fft.buffer)).Free()
	}
	// TODO destroy, free the buffer
}

func (fft *FFTPlan5) Forward(in, out *Array) {
	AssertMsg(in.size4D[0] == 1, "1")
	AssertMsg(out.size4D[0] == 1, "2")
	CheckSize(in.size3D, fft.dataSize[:])
	CheckSize(out.size3D, fft.outputSize[:])
    Start("FW_total")
	if NDevice() == 1 { //  single-gpu implementation

		// 		fmt.Println("single GPU used")

		fftZ1Dev := fft.fftZ1Dev

		// zero padding, all FFTs are in-place
		CopyPad3D(out, in)
		//   fmt.Println("")
		//   fmt.Println("zero padding:", out.LocalCopy().Array)

		// FFT in z-direction
		offset := ((fft.logicSize[2]) + 2) * fft.logicSize[1]
		for i := 0; i < fft.dataSize[0]; i++ { // TODO check if streams per plane are faster
			fftZ1Dev[i].PointTo(out, i*offset)
			ptr := uintptr(fftZ1Dev[i].pointer[0])
			fft.planZ_FW[0].ExecR2C(ptr, ptr)
		}
		fft.Sync() //  Is this required?
		//   fmt.Println("")
		//   fmt.Println("FFTZ:", out.LocalCopy().Array)

		// FFT in y-direction
		for i := 0; i < fft.dataSize[0]; i++ { // TODO check if streams per plane are faster
			ptr := uintptr(fftZ1Dev[i].pointer[0])
			fft.planY[0].ExecC2C(ptr, ptr, cufft.FORWARD) //FFT in y-direction
		}
		fft.Sync() //  Is this required?
		//   fmt.Println("")
		//   fmt.Println("FFTY:", out.LocalCopy().Array)

		// FFT in x-direction
		if fft.logicSize[0] > 1 {
			fft.planX[0].ExecC2C(uintptr(out.pointer[0]), uintptr(out.pointer[0]), cufft.FORWARD) //FFT in x-direction
		}
		fft.Sync() //  Is this required?
	} else { //  multi-gpu implementation

		//   fmt.Println("FORWARD FFT")
		//   fmt.Println("")

		// shorthand and define ghost arrays ----------------------------
		buffer := &(fft.buffer)
		padZ := &(fft.padZ)
		padZ.PointTo(out, 0)
		fftZbuffer := &(fft.fftZbuffer)
		fftZbuffer.PointTo(buffer, 0)
		transp1 := &(fft.transp1)
		transp1.PointTo(out, 0)
		chunks := fft.chunks          // not sure if chunks[0] copies the struct...
		for dev := range _useDevice { // source device
			chunks[dev].PointTo(buffer, chunks[dev].Len())
		}
		transp2 := &(fft.transp2)
		transp2.PointTo(out, 0)

		dataSize := fft.dataSize
		logicSize := fft.logicSize
		NDev := NDevice()
		// -------------------------------------------------------------

		  fmt.Println("in: ", in.LocalCopy().Array)

    fft.Sync()

		// @@@@@@@@ SYNCHRONIZATION: FROM THIS POINT ON, ALL IS DONE ON THE COMPLETE DATA SET @@@@@@@@
		Start("FW_pad")
		CopyPadZAsync(padZ, in, fft.Stream)
		Stop("FW_pad")

		Start("FW_fftZ")
		for dev := range _useDevice {
			setDevice(_useDevice[dev])
			fft.planZ_FW[dev].ExecR2C(uintptr(padZ.pointer[dev]), uintptr(fftZbuffer.pointer[dev])) // is this really async?
		}
		fft.Sync()
    Stop("FW_fftZ")
    
    fft.Sync()
    fmt.Println("")
    fmt.Println("fftZbuffer: ", fftZbuffer.LocalCopy().Array)

		// @@@@@@@@ SYNCHRONIZATION: FROM THIS POINT ON, ALL IS DONE ON PLANES @@@@@@@@
		Start("FW_Transpose")
		//  TransposeComplexYZPart(transp1, fftZbuffer) // fftZ!
		TransposeComplexYZPartAsync(transp1, fftZbuffer, fft.Stream) // fftZ!
		Stop("FW_Transpose")
    
    fft.Sync()
    fmt.Println("")
    fmt.Println("transp1: ", transp1.LocalCopy().Array)

    Start("FW_zero")
    ZeroArrayAsync(transp2, fft.Stream)
    Stop("FW_zero")

		// copy chunks, cross-device
		Start("FW_copy")
//    chunkLineBytes := int64(chunks[0].partSize[2]) * SIZEOF_FLOAT // one plane 
    chunkLineBytes := int64(dataSize[1]*2/NDev) * SIZEOF_FLOAT // one line 
		Assert(dataSize[1]%NDev == 0)
		Assert(logicSize[2]%NDev == 0)

    srcPlaneN := transp1.partSize[1] * transp1.partSize[2]
    dstPlaneN := (logicSize[2]/NDev/2+1) * logicSize[1]*2
    for dev := range _useDevice { // source device
			for c := range chunks { // source chunk
        fmt.Println("jmax: ",(logicSize[2]/2/NDev +1) )
				for i := 0; i < dataSize[0]; i++ { // only memcpys in this loop
          for j :=0; j< (logicSize[2]/NDev/2 +1); j++{
//           for j :=0; j< 2; j++{
//            srcOffset := i*srcPlaneN + c*((dataSize[1]/NDev)*(logicSize[2]/NDev)) + j*(dataSize[1]/NDev*2)
            srcOffset := i*srcPlaneN + c*((dataSize[1]*2/NDev)*(logicSize[2]/2/NDev)+1) + j*(dataSize[1]/NDev*2)
            src := cu.DevicePtr(ArrayOffset(uintptr(transp1.pointer[dev]), srcOffset))

//             dstOffset := i*dstPlaneN + j*logicSize[1]*2 + dev*chunks[0].partSize[2]
            dstOffset := i*dstPlaneN + j*logicSize[1]*2 + dev *(dataSize[1]*2/NDev)
            dst := cu.DevicePtr(ArrayOffset(uintptr(transp2.pointer[c]), dstOffset))
//             dst := cu.DevicePtr(ArrayOffset(uintptr(transp1.pointer[dev]), srcOffset))

            cu.MemcpyDtoDAsync(dst, src, chunkLineBytes, fft.Stream[dev])
          }
				}
			}
		}
		Stop("FW_copy")
    
    fft.Sync()
    fmt.Println("")
    fmt.Println("transp2:", transp2.LocalCopy().Array)

    
// 		Start("FW_insertBlockZ")
// 		for c := range chunks {
// 			InsertBlockZAsync(transp2, &(chunks[c]), c, fft.Stream)
// 		}
// 		fft.Sync()
//     Stop("FW_insertBlockZ")

    
    
		// @@@@@@@@ SYNCHRONIZATION: FROM THIS POINT ALL IS DONE ON THE COMPLETE DATA SET @@@@@@@@
		Start("FW_fftY")
		for dev := range _useDevice {
			setDevice(_useDevice[dev])
			fft.planY[dev].ExecC2C(uintptr(transp2.pointer[dev]), uintptr(out.pointer[dev]), cufft.FORWARD) //FFT in y-direction
		}
		   fft.Sync()    // Can probably deleted.  All FFTs on one device should be finished before going further.
		Stop("FW_fftY")

		// FFT X
		if logicSize[0] > 1 {
			Start("FW_fftX")
			for dev := range _useDevice {
        setDevice(_useDevice[dev])
				fft.planX[dev].ExecC2C(uintptr(out.pointer[dev]), uintptr(out.pointer[dev]), cufft.FORWARD) //FFT in x-direction
			}
			    fft.Sync()    // Can probably deleted.  All FFTs on one device should be finished before going further.
			Stop("FW_fftX")
		}
		/*  fmt.Println("")
		    fmt.Println("out:", out.LocalCopy().Array)*/
    Start("lastsync")
		fft.Sync()
    Stop("lastsync")
	}

	/*  fmt.Println("")
	fmt.Println("out:", out.LocalCopy().Array)*/
	Stop("FW_total")
}

func (fft *FFTPlan5) Inverse(in, out *Array) {

	/*  fmt.Println("")
	fmt.Println("")
	fmt.Println("INVERSE FFT")
	fmt.Println("")
	fmt.Println("in:", in.LocalCopy().Array)
	*/

  //Start("INV_total")
	if NDevice() == 1 { //  single-gpu implementation

		fftZ1Dev := fft.fftZ1Dev

		// FFT in x-direction
		if fft.logicSize[0] > 1 {
			fft.planX[0].ExecC2C(uintptr(in.pointer[0]), uintptr(in.pointer[0]), cufft.INVERSE) //FFT in x-direction
		}
		fft.Sync() //  Is this required?

		// FFT in y-direction
		offset := ((fft.logicSize[2]) + 2) * fft.logicSize[1]
  
		for i := 0; i < fft.dataSize[0]; i++ { // TODO check if streams per plane are faster
			fftZ1Dev[i].PointTo(in, i*offset)
			ptr := uintptr(fftZ1Dev[i].pointer[0])
			fft.planY[0].ExecC2C(ptr, ptr, cufft.INVERSE) //FFT in y-direction
		}
		fft.Sync() //  Is this required?
		//   fmt.Println("")
		//   fmt.Println("inv FFTy:", in.LocalCopy().Array)

		// FFT in z-direction
		for i := 0; i < fft.dataSize[0]; i++ {
			ptr := uintptr(fftZ1Dev[i].pointer[0])
			fft.planZ_INV[0].ExecC2R(ptr, ptr)
		}
		fft.Sync() //  Is this required?
		//   fmt.Println("")
		//   fmt.Println("before unpadding:", in.LocalCopy().Array)

		// extracting data
		CopyPad3D(out, in)

	} else { //  single-gpu implementation

		// shorthand
		buffer := &(fft.buffer)
		padZ := &(fft.padZ)
		padZ.PointTo(in, 0)
		fftZbuffer := &(fft.fftZbuffer)
		fftZbuffer.PointTo(buffer, 0)
		transp1 := &(fft.transp1)
		transp1.PointTo(in, 0)
		chunks := fft.chunks          // not sure if chunks[0] copies the struct...
		for dev := range _useDevice { // source device
			chunks[dev].PointTo(buffer, chunks[dev].Len())
		}
		transp2 := &(fft.transp2)
		transp2.PointTo(in, 0)

		dataSize := fft.dataSize
		logicSize := fft.logicSize
		NDev := NDevice()

		// FFT X
		if logicSize[0] > 1 {
			    //Start("INV_fftX")
			//     fmt.Println("")
			for dev := range _useDevice {
				setDevice(_useDevice[dev])
				fft.planX[dev].ExecC2C(uintptr(in.pointer[dev]), uintptr(in.pointer[dev]), cufft.INVERSE) //FFT in x-direction
			}
			fft.Sync()
			    //Stop("INV_fftX")
			/*    fmt.Println("")
			      fmt.Println("fftx:", in.LocalCopy().Array)*/
		}

		// FFT Y
		  //Start("INV_fftY")
		for dev := range _useDevice {
			setDevice(_useDevice[dev])
			fft.planY[dev].ExecC2C(uintptr(in.pointer[dev]), uintptr(transp2.pointer[dev]), cufft.INVERSE) //FFT in y-direction
		}
		fft.Sync()
		  //Stop("INV_fftY")
		//   fmt.Println("")
		//   fmt.Println("ffty:", transp2.LocalCopy().Array)


		for c := range chunks {
			ExtractBlockZ(&(chunks[c]), transp2, c)
		}
    fft.Sync()
		//   fmt.Println("")
		//   fmt.Println("extract:", transp2.LocalCopy().Array)

    //Start("INV_copy")
		// copy chunks, cross-device
		chunkPlaneBytes := int64(chunks[0].partSize[1]*chunks[0].partSize[2]) * SIZEOF_FLOAT // one plane 
		for dev := range _useDevice {                                                        // source device
			for c := range chunks {
				for i := 0; i < dataSize[0]; i++ { // only memcpys in this loop
					srcPlaneN := chunks[0].partSize[1] * chunks[0].partSize[2] //fmt.Println("dstPlaneN:", dstPlaneN)//seems OK
					srcOffset := i * srcPlaneN
					src := cu.DevicePtr(ArrayOffset(uintptr(chunks[dev].pointer[c]), srcOffset))

					dstPlaneN := transp1.partSize[1] * transp1.partSize[2] //fmt.Println("srcPlaneN:", srcPlaneN)//seems OK
					dstOffset := i*dstPlaneN + c*((dataSize[1]/NDev)*(logicSize[2]/NDev))
					dst := cu.DevicePtr(ArrayOffset(uintptr(transp1.pointer[dev]), dstOffset))

					// must be done plane by plane
					cu.MemcpyDtoDAsync(dst, src, chunkPlaneBytes, fft.Stream[dev]) // chunkPlaneBytes for plane-by-plane
				}
			}
		}
    fft.Sync()
		//   fmt.Println("")
		//   fmt.Println("copy:", transp1.LocalCopy().Array)
    //Stop("INV_copy")

    //Start("INV_transp")
		TransposeComplexYZPart_inv(fftZbuffer, transp1) // fftZ!
		//   fmt.Println("")
		//   fmt.Println("transpose:", padZ.LocalCopy().Array)
    //Stop("INV_transp")

    fft.Sync()
		// fft Z
    //Start("INV_FFTZ")
		for dev := range _useDevice {
			setDevice(_useDevice[dev])
			fft.planZ_INV[dev].ExecC2R(uintptr(fftZbuffer.pointer[dev]), uintptr(padZ.pointer[dev])) // is this really async?
		}
		fft.Sync()
		//   fmt.Println("")
		//   fmt.Println("fftZ:", padZ.LocalCopy().Array)
    //Stop("INV_FFTZ")
    
    //Start("INV_unpad")
		CopyPadZ(out, padZ)
    //Stop("INV_unpad")
    fft.Sync()
	}
	// 	fmt.Println("")
	// 	fmt.Println("out:", out.LocalCopy().Array)
	//Stop("INV_total")

}
