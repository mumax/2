//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Combined demag+exchange module.
// Demag and exchange field are calculated in one single convolution.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	"mumax/host"
)

// Register this module
func init() {
	RegisterModule("demagexch", "Provides combined magnetostatic + exchange field", LoadDemagExch)
}

func LoadDemagExch(e *Engine) {

	// dependencies
	LoadHField(e)
	LoadMagnetization(e)
	Aex := e.AddNewQuant("Aex", SCALAR, VALUE, Unit("J/m"), "exchange coefficient") // here it has to be a value.

	m := e.Quant("m")
	MSat := e.Quant("MSat")

	CPUONLY := true
	// Size of all kernels (not FFT'd)
	kernelSize := padSize(e.GridSize(), e.Periodic())

	// demag kernel 
	demagAcc := e.AddNewQuant("demag_acc", SCALAR, VALUE, Unit(""), "demag field accuracy")
	demagAcc.SetScalar(8)
	demagAcc.SetVerifier(Uint)
	demagKern := NewQuant("kern_d", SYMMTENS, kernelSize, FIELD, Unit(""), CPUONLY, "reduced demag kernel (/Msat)")
	e.AddQuant(demagKern)
	e.Depends("kern_d", "demag_acc")
	demagKern.SetUpdater(newDemagKernUpdater(demagKern))

	// exch kernel 
	exchKern := NewQuant("kern_ex", SYMMTENS, kernelSize, FIELD, Unit("/m2"), CPUONLY, "reduced exchange kernel (Laplacian)")
	e.AddQuant(exchKern)
	//exRange := e.AddNewQuant("ex_range", SCALAR, VALUE, Unit("cells"), "exchange interaction range in cells; 1:nearest, 2:next-nearest,...")
	//exRange.SetVerifier(PosInt)
	//exRange.SetScalar(1)
	//e.Depends("kern_ex", "ex_range")
	exchKern.SetUpdater(newExchKernUpdater(exchKern))

	// demag+exchange kernel
	dexKern := NewQuant("Kern_dex", SYMMTENS, kernelSize, FIELD, Unit("A/m"), CPUONLY, "demag+exchange kernel")
	e.AddQuant(dexKern)
	e.Depends("Kern_dex", "kern_d", "kern_ex", "Aex", "MSat")
	dexKern.SetUpdater(newDexKernUpdater(dexKern, demagKern, exchKern, MSat, Aex))

	// fft kernel quant
	fftOutSize := gpu.FFTOutputSize(kernelSize)
	fftOutSize[2] /= 2 // only real parts are stored
	fftKern := NewQuant("~kern_dex", SYMMTENS, fftOutSize, FIELD, Unit("A/m"), false, "FFT demag+exchange kernel")
	e.AddQuant(fftKern)
	e.Depends("~kern_dex", "kern_dex")
	fftKern.SetUpdater(newFftKernUpdater(fftKern, dexKern))

	// demag+exchange field quant
	Hdex := e.AddNewQuant("H_dex", VECTOR, FIELD, Unit("A/m"), "demag+exchange field")
	e.Depends("H_dex", "m", "~Kern_dex")
	Hdex.SetUpdater(newHDexUpdater(Hdex, m, fftKern))

	// add H_dex to total H
	hfield := e.Quant("H")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_dex")
}

//____________________________________________________________________ demag kernel

// Update demag kernel (cpu)
type demagKernUpdater struct {
	demagKern *Quant // that's me!
}

func newDemagKernUpdater(demagKern *Quant) Updater {
	u := new(demagKernUpdater)
	u.demagKern = demagKern
	return u
}

// Update demag kernel (cpu)
func (u *demagKernUpdater) Update() {
	e := GetEngine()
	kernsize := padSize(e.GridSize(), e.Periodic())
	accuracy := int(e.Quant("demag_acc").Scalar())
	// TODO: wisdom
	Log("Calculating demag kernel, may take a moment...")
	FaceKernel6(kernsize, e.CellSize(), accuracy, e.Periodic(), u.demagKern.Buffer())
}

//____________________________________________________________________ exchange kernel

// Update exchange kernel (cpu)
type exchKernUpdater struct {
	exchKern *Quant // that's me!
}

func newExchKernUpdater(exchKern *Quant) Updater {
	u := new(exchKernUpdater)
	u.exchKern = exchKern
	return u
}

// Update exch kernel (cpu)
func (u *exchKernUpdater) Update() {
	e := GetEngine()
	kernsize := padSize(e.GridSize(), e.Periodic())
	//exRange := e.Quant("ex_range").Scalar()
	// Fast, so no wisdom needed here
	Exch6NgbrKernel(kernsize, e.CellSize(), u.exchKern.Buffer()) //, exRange)
}
//____________________________________________________________________ demag+exchange kernel

// Update demag+exchange kernel (cpu)
type dexKernUpdater struct {
	dexKern                        *Quant // that's me!
	demagKern, exchKern, MSat, Aex *Quant // my dependencies
}

func newDexKernUpdater(dexKern, demagKern, exchKern, MSat, Aex *Quant) Updater {
	// TODO: verify Aex space-independent
	CheckSize(dexKern.Size3D(), demagKern.Size3D())
	CheckSize(dexKern.Size3D(), exchKern.Size3D())
	return &dexKernUpdater{dexKern, demagKern, exchKern, MSat, Aex}
}

// Update demag+exchange kernel (cpu)
func (u *dexKernUpdater) Update() {
	Debug("Update demagexch")

	dex := u.dexKern.Buffer().List
	demag := u.demagKern.Buffer().List
	exch := u.exchKern.Buffer().List
	MSat := u.MSat.Multiplier()[0] // Msat may be mask.
	Aex := u.Aex.Scalar()
	for i := range dex {
		// dex = MSat * demag + 2A/µ0MSat * laplacian
		dex[i] = float32(MSat*float64(demag[i]) + ((2*Aex)/(Mu0*MSat))*float64(exch[i]))
	}
}

//_____________________________________________________________________ fftkern

// Holds any transformed kernel 
// as well as the convolution plan that goes with it.
type fftKernUpdater struct {
	fftKern *Quant       // that's me!
	kern    *Quant       //  my dependencies
	conv    gpu.ConvPlan // TODO: move gpu.ConvPlan into engine?
}

func newFftKernUpdater(fftKern, kern *Quant) Updater {
	u := new(fftKernUpdater)
	u.fftKern = fftKern
	u.kern = kern
	return u
}

func (u *fftKernUpdater) Update() {
	Debug("Update fftKern")
	dataSize := GetEngine().GridSize()
	kernel := make([]*host.Array, 6)
	for i := range kernel {
		kernel[i] = (u.kern.Buffer().Component(i))
	}
	u.conv.Init(dataSize, kernel, u.fftKern.Array())
	u.conv.SelfTest()
}

//____________________________________________________________________ H_dex

// Updates the demag+exchange field in one single convolution
type hDexUpdater struct {
	Hdex, m, fftDexKern *Quant
	conv                *gpu.ConvPlan // points to fftKernDex.updater.conv
}

func newHDexUpdater(Hdex, m, fftDexKern *Quant) Updater {
	u := new(hDexUpdater)
	u.Hdex = Hdex
	u.m = m
	u.fftDexKern = fftDexKern
	u.conv = &(fftDexKern.Updater().(*fftKernUpdater).conv)
	return u
}

func (u *hDexUpdater) Update() {
	u.conv.Convolve(u.m.Array(), u.Hdex.Array())
}
