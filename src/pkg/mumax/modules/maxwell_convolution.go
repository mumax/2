//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// 14-input, 6 output convolution for solving the general Maxwell equations.
// This implementation makes a trade-off: use less memory (good!) but more memory bandwidth (bad).
// The implementation is thus optimized for low memory usage, not for absolute top speed.
// Speed could be gained when a blocking memory recycler is in place.
//
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	"mumax/host"
	"fmt"
	"unsafe"
)


// Full Maxwell Electromagnetic field solver.
// TODO: magnetic charge gives H, not B, need M
type MaxwellPlan struct {
	dataSize     [3]int           // Size of the (non-zero) input data block (engine.GridSize)
	logicSize    [3]int           // Non-transformed kernel size >= dataSize (due to zeropadding)
	fftKernSize  [3]int           // transformed kernel size, non-redundant parts only
	kern         [3]*host.Array   // Real-space kernels for charge, dipole, rotor
	fftKern      [7][3]*gpu.Array // transformed kernel's non-redundant parts (only real or imag parts, or nil)
	fftMul       [7][3]complex128 // multipliers for kernel
	fftBuffer    gpu.Array        // transformed input data
	fftOut       gpu.Array        // transformed output data (3-comp)
	fftPlan      gpu.FFTInterface // transforms input/output data
	Einput       [7]*gpu.Array    // input quantities for electric field (rho, Px, Py, Pz, 0, 0, 0)
	Binput       [7]*gpu.Array    // input quantities for magnetic field (rhoB, mx, my, mz, jx, jy, jz)
	EInMul       [7]complex128    // E input multipliers (epsillon0 etc)
	BInMul       [7]complex128    // B input multipliers (mu0 etc)
	fftE1, fftE2 *gpu.Array       // previous FFT E fields for time derivative 
	fftB1, fftB2 *gpu.Array       // previous FFT B fields for time derivative 
	time1, time2 float64          // time of previous fields for derivative
	BExt, EExt   *Quant           // external B/E field
	E, B         *Quant           // E/B fields
}

const (
	CHARGE = 0
	DIPOLE = 1
	ROTOR  = 2
)


func (plan *MaxwellPlan) Init(dataSize, logicSize []int) {
	Assert(len(dataSize) == 3)
	Assert(len(logicSize) == 3)

	//plan.Free() // must not leak memory on 2nd init. // TODO

	// init size
	copy(plan.dataSize[:], dataSize)
	copy(plan.logicSize[:], logicSize)

	// init fft
	fftOutputSize := gpu.FFTOutputSize(logicSize)
	plan.fftBuffer.Init(1, fftOutputSize, gpu.DO_ALLOC) // TODO: recycle
	plan.fftOut.Init(3, fftOutputSize, gpu.DO_ALLOC)    // TODO: recycle
	plan.fftPlan = gpu.NewDefaultFFT(dataSize, logicSize)

	// init fftKern
	copy(plan.fftKernSize[:], gpu.FFTOutputSize(logicSize))
	plan.fftKernSize[2] = plan.fftKernSize[2] / 2 // store only non-redundant parts
}


func (plan *MaxwellPlan) AddCoulomb() {
	if plan.kern[CHARGE] == nil {
		plan.LoadChargeKernel()
	}

}

func (plan *MaxwellPlan) LoadChargeKernel() {
	e := GetEngine()
	const (
		CPUONLY = true
		GPU     = false
	)
	// Add the kernel as orphan quant, so we can output it.
	// TODO: do not add to engine if debug is off?
	quant := NewQuant("kern_el", VECTOR, plan.logicSize[:], FIELD, Unit(""), CPUONLY, "reduced electrostatic kernel")
	e.AddQuant(quant)
	kern := quant.Buffer()
	PointKernel(plan.logicSize[:], e.CellSize(), e.Periodic(), kern)
	plan.kern[CHARGE] = kern

	plan.LoadKernel(kern, 0, DIAGONAL, PUREIMAG)
}

//
//func (conv *Conv73Plan) Convolve(in []*gpu.Array, out *gpu.Array) {
//	fftBuf := &conv.fftBuffer
//	fftOut := &conv.fftOut
//	fftOut.Zero()
//	for i := 0; i < 7; i++ {
//		if in[i] == nil {
//			continue
//		}
//		conv.ForwardFFT(in[i])
//		//fmt.Println("conv.fftBuffer", i, conv.fftBuffer.LocalCopy().Array, "\n")
//		for j := 0; j < 3; j++ {
//			if conv.fftKern[i][j] == nil {
//				continue
//			}
//			//fmt.Println("conv.fftKern", i, j, conv.fftKern[i][j].LocalCopy().Array, "\n")
//			// Point-wise kernel multiplication
//			CMaddAsync(&fftOut.Comp[j], conv.fftMul[i][j], conv.fftKern[i][j], fftBuf, fftOut.Stream)
//			fftOut.Stream.Sync()
//			//fmt.Println("conv.fftOut", j, conv.fftOut.Comp[j].LocalCopy().Array, "\n")
//		}
//	}
//	conv.InverseFFT(out)
//	//fmt.Println("conv out", out.LocalCopy().Array, "\n")
//}


//// Loads a sub-kernel at position pos in the 3x7 global kernel matrix.
//// The symmetry and real/imaginary/complex properties are taken into account to reduce storage.
func (plan *MaxwellPlan) LoadKernel(kernel *host.Array, pos int, matsymm int, realness int) {
	//Assert(kernel.NComp() == 9) // full tensor
	if kernel.NComp() == 9 {
		Assert(matsymm == MatrixSymmetry(kernel))
	}
	Assert(matsymm == SYMMETRIC || matsymm == ANTISYMMETRIC || matsymm == NOSYMMETRY || matsymm == DIAGONAL)

	//if FFT'd kernel is pure real or imag, 
	//store only relevant part and multiply by scaling later
	scaling := [3]complex128{complex(1, 0), complex(0, 1), complex(0, 0)}[realness]
	Debug("scaling=", scaling)

	// FFT input on GPU
	logic := plan.logicSize[:]
	devIn := gpu.NewArray(1, logic)
	defer devIn.Free()

	// FFT output on GPU
	devOut := gpu.NewArray(1, gpu.FFTOutputSize(logic))
	defer devOut.Free()
	fullFFTPlan := gpu.NewDefaultFFT(logic, logic)
	defer fullFFTPlan.Free()

	// FFT all components
	for k := 0; k < 9; k++ {
		i, j := IdxToIJ(k) // fills diagonal first, then upper, then lower

		// ignore off-diagonals of vector (would go out of bounds)
		if k > ZZ && matsymm == DIAGONAL {
			Debug("break", TensorIndexStr[k], "(off-diagonal)")
			break
		}

		// elements of diagonal kernel are stored in one column
		if matsymm == DIAGONAL {
			i = 0
		}

		// clear data first
		AssertMsg(plan.fftKern[i+pos][j] == nil, "I'm afraid I can't let you overwrite that")
		AssertMsg(plan.fftMul[i+pos][j] == 0, "Likewise")

		// ignore zeros
		if k < kernel.NComp() && IsZero(kernel.Comp[k]) {
			Debug("kernel", TensorIndexStr[k], " == 0")
			continue
		}

		// auto-fill lower triangle if possible
		if k > XY {
			if matsymm == SYMMETRIC {
				plan.fftKern[i+pos][j] = plan.fftKern[j+pos][i]
				plan.fftMul[i+pos][j] = plan.fftMul[j+pos][i]
				continue
			}
			if matsymm == ANTISYMMETRIC {
				plan.fftKern[i+pos][j] = plan.fftKern[j+pos][i]
				plan.fftMul[i+pos][j] = -plan.fftMul[j+pos][i]
				continue
			}
		}

		// calculate FFT of kernel element
		Debug("use", TensorIndexStr[k])
		devIn.CopyFromHost(kernel.Component(k))
		fullFFTPlan.Forward(devIn, devOut)
		hostOut := devOut.LocalCopy()

		// extract real or imag parts
		hostFFTKern := extract(hostOut, realness)
		rescale(hostFFTKern, 1/float64(gpu.FFTNormLogic(logic)))
		plan.fftKern[i+pos][j] = gpu.NewArray(1, hostFFTKern.Size3D)
		plan.fftKern[i+pos][j].CopyFromHost(hostFFTKern)
		plan.fftMul[i+pos][j] = scaling
	}

	// debug
	var dbg [7][3]string
	for i := range dbg {
		for j := range dbg[i] {
			dbg[i][j] = fmt.Sprint(unsafe.Pointer(plan.fftKern[i][j]), "*", plan.fftMul[i][j])
		}
	}
	Debug("maxwell convplan kernel:", dbg)
}


func IsZero(array []float32) bool {
	for _, x := range array {
		if x != 0 {
			return false
		}
	}
	return true
}


// arr[i] *= scale
func rescale(arr *host.Array, scale float64) {
	list := arr.List
	for i := range list {
		list[i] = float32(float64(list[i]) * scale)
	}
}

// matrix symmetry
const (
	NOSYMMETRY    = 0  // Kij independent of Kji
	SYMMETRIC     = 1  // Kij = Kji
	DIAGONAL      = 2  // also used for vector
	ANTISYMMETRIC = -1 // Kij = -Kji
)

// Detects matrix symmetry.
// returns NOSYMMETRY, SYMMETRIC, ANTISYMMETRIC 
func MatrixSymmetry(matrix *host.Array) int {
	AssertMsg(matrix.NComp() == 9, "MatrixSymmetry NComp")
	symm := true
	asymm := true
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			idx1 := TensorIdx[i][j]
			idx2 := TensorIdx[j][i]
			comp1 := matrix.Comp[idx1]
			comp2 := matrix.Comp[idx2]
			for x := range comp1 {
				if comp1[x] != comp2[x] {
					symm = false
					if !asymm {
						break
					}
				}
				if comp1[x] != -comp2[x] {
					asymm = false
					if !symm {
						break
					}
				}
			}
		}
	}
	if symm {
		return SYMMETRIC // also covers all zeros
	}
	if asymm {
		return ANTISYMMETRIC
	}
	return NOSYMMETRY
}


// data realness
const (
	PUREREAL = 0 // data is purely real
	PUREIMAG = 1 // data is purely complex
	COMPLEX  = 2 // data is full complex number
)


func (plan *MaxwellPlan) Free() {
	// TODO
}

//
//// 	INTERNAL
//// Sparse transform all 3 components.
//// (FFTPlan knows about zero padding etc)
//func (conv *Conv73Plan) ForwardFFT(in *Array) {
//	Assert(conv.fftBuffer.NComp() == in.NComp())
//	//for c := range in.Comp {
//	conv.fftPlan.Forward(in, &conv.fftBuffer)
//	//}
//}
//
//// 	INTERNAL
//// Sparse backtransform
//// (FFTPlan knows about zero padding etc)
//func (conv *Conv73Plan) InverseFFT(out *Array) {
//	Assert(conv.fftOut.NComp() == out.NComp())
//	for c := range out.Comp {
//		conv.fftPlan.Inverse(&conv.fftOut.Comp[c], &out.Comp[c])
//	}
//}
//
//func (conv *Conv73Plan) SelfTest() {
//	Debug("FFT self-test")
//	rng := rand.New(rand.NewSource(0))
//	size := conv.dataSize[:]
//
//	in := NewArray(1, size)
//	defer in.Free()
//	arr := in.LocalCopy()
//	a := arr.List
//	for i := range a {
//		a[i] = 2*rng.Float32() - 1
//		if a[i] == 0 {
//			a[i] = 1
//		}
//	}
//	in.CopyFromHost(arr)
//
//	out := NewArray(1, size)
//	defer out.Free()
//
//	conv.ForwardFFT(in)
//	conv.InverseFFT(out)
//
//	b := out.LocalCopy().List
//	norm := float32(1 / float64(FFTNormLogic(conv.logicSize[:])))
//	var maxerr float32
//	for i := range a {
//		if Abs32(a[i]-b[i]*norm) > maxerr {
//			maxerr = Abs32(a[i] - b[i]*norm)
//		}
//	}
//	Debug("FFT max error:", maxerr)
//	if maxerr > 1e-3 {
//		panic(BugF("FFT self-test failed, max error:", maxerr, "\nPlease use a different grid size of FFT type."))
//	}
//	runtime.GC()
//}


// Extract real or imaginary parts, copy them from src to dst.
// In the meanwhile, check if the other parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
// real_imag = 0: real parts
// real_imag = 1: imag parts
func extract(src *host.Array, realness int) *host.Array {
	if realness == COMPLEX {
		return src
	}

	sx := src.Size3D[X]
	sy := src.Size3D[Y]
	sz := src.Size3D[Z] / 2 // only real/imag parts
	dst := host.NewArray(src.NComp(), []int{sx, sy, sz})

	dstList := dst.List
	srcList := src.List

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maxbad := float32(0.)
	maxgood := float32(0.)
	other := 1 - realness
	for i := range dstList {
		dstList[i] = srcList[2*i+realness]
		if Abs32(srcList[2*i+other]) > maxbad {
			maxbad = Abs32(srcList[2*i+other])
		}
		if Abs32(srcList[2*i+realness]) > maxgood {
			maxgood = Abs32(srcList[2*i+realness])
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	Debug("FFT Kernel max part", realness, ":", maxgood)
	Debug("FFT Kernel max part", other, ":", maxbad)
	Debug("FFT Kernel max bad/good part=", maxbad/maxgood)
	if maxbad/maxgood > 1e-5 { // TODO: is this reasonable?
		panic(BugF("FFT Kernel max bad/good part=", maxbad/maxgood))
	}
	return dst
}
