//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Author: Arne Vansteenkiste

import (
	//. "mumax/common"
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("current", "Electrical current", LoadCurrent)
}

func LoadCurrent(e *Engine) {

	e.AddNewQuant("j", VECTOR, FIELD, Unit("A/m2"), "electrical current density")
	e.AddNewQuant("rho", SCALAR, FIELD, Unit("C/m3"), "electrical charge density")
	e.AddNewQuant("E", VECTOR, FIELD, Unit("V/m"), "electrical field")
	e.AddNewQuant("sigma", SCALAR, MASK, Unit("1/Ohm*m"), "electrical conductivity")
	e.AddNewQuant("diff_rho", SCALAR, FIELD, Unit("C/m3s"), "time derivative of rho")

	e.Depends("E", "rho")
	e.Depends("j", "E", "sigma")
	e.Depends("diff_rho", "j")

	e.AddPDE1("rho", "diff_rho")
}
//	// dependencies
//	e.LoadModule("hfield")
//	e.LoadModule("magnetization")
//	e.AddQuant("Aex", SCALAR, VALUE, Unit("J/m"), "exchange coefficient") // here it has to be a value.
//
//	m := e.Quant("m")
//	MSat := e.Quant("MSat")
//	Aex := e.Quant("Aex")
//
//	CPUONLY := true
//	// Size of all kernels (not FFT'd)
//	kernelSize := padSize(e.GridSize(), e.Periodic())
//
//	// demag kernel 
//	demagKern := newQuant("kern_d", SYMMTENS, kernelSize, FIELD, Unit(""), CPUONLY, "reduced demag kernel (/Msat)")
//	e.addQuant(demagKern)
//	demagKern.SetUpdater(newDemagKernUpdater(demagKern))
//
//	// exch kernel 
//	exchKern := newQuant("kern_ex", SYMMTENS, kernelSize, FIELD, Unit("/m2"), CPUONLY, "reduced exchange kernel (Laplacian)")
//	e.addQuant(exchKern)
//	exchKern.SetUpdater(newExchKernUpdater(exchKern))
//
//	// demag+exchange kernel
//	dexKern := newQuant("Kern_dex", SYMMTENS, kernelSize, FIELD, Unit("A/m"), CPUONLY, "demag+exchange kernel")
//	e.addQuant(dexKern)
//	e.Depends("Kern_dex", "kern_d", "kern_ex", "Aex", "MSat")
//	dexKern.SetUpdater(newDexKernUpdater(dexKern, demagKern, exchKern, MSat, Aex))
//
//	// fft kernel quant
//	fftOutSize := gpu.FFTOutputSize(kernelSize)
//	fftOutSize[2] /= 2 // only real parts are stored
//	fftKern := newQuant("~kern_dex", SYMMTENS, fftOutSize, FIELD, Unit("A/m"), false, "FFT demag+exchange kernel")
//	e.addQuant(fftKern)
//	e.Depends("~kern_dex", "kern_dex")
//	fftKern.SetUpdater(newFftKernUpdater(fftKern, dexKern))
//
//	// demag+exchange field quant
//	e.AddQuant("H_dex", VECTOR, FIELD, Unit("A/m"), "demag+exchange field")
//	e.Depends("H_dex", "m", "~Kern_dex")
//	Hdex := e.Quant("H_dex")
//	Hdex.SetUpdater(newHDexUpdater(Hdex, m, fftKern))
//
//	// add H_dex to total H
//	hfield := e.Quant("H")
//	sum := hfield.updater.(*SumUpdater)
//	sum.AddParent("H_dex")
//}
//
////____________________________________________________________________ demag kernel
//
//// Update demag kernel (cpu)
//type demagKernUpdater struct {
//	demagKern *Quant // that's me!
//}
//
//func newDemagKernUpdater(demagKern *Quant) Updater {
//	u := new(demagKernUpdater)
//	u.demagKern = demagKern
//	return u
//}
//
//// Update demag kernel (cpu)
//func (u *demagKernUpdater) Update() {
//	e := GetEngine()
//	kernsize := padSize(e.GridSize(), e.Periodic())
//	accuracy := 8
//	// TODO: wisdom
//	Log("Calculating demag kernel, may take a moment...")
//	FaceKernel6(kernsize, e.CellSize(), accuracy, e.Periodic(), u.demagKern.Buffer())
//}
//
////____________________________________________________________________ exchange kernel
//
//// Update exchange kernel (cpu)
//type exchKernUpdater struct {
//	exchKern *Quant // that's me!
//}
//
//func newExchKernUpdater(exchKern *Quant) Updater {
//	u := new(exchKernUpdater)
//	u.exchKern = exchKern
//	return u
//}
//
//// Update exch kernel (cpu)
//func (u *exchKernUpdater) Update() {
//	e := GetEngine()
//	kernsize := padSize(e.GridSize(), e.Periodic())
//	// Fast, so no wisdom needed here
//	Exch6NgbrKernel(kernsize, e.CellSize(), u.exchKern.Buffer())
//}
////____________________________________________________________________ demag+exchange kernel
//
//// Update demag+exchange kernel (cpu)
//type dexKernUpdater struct {
//	dexKern                        *Quant // that's me!
//	demagKern, exchKern, MSat, Aex *Quant // my dependencies
//}
//
//func newDexKernUpdater(dexKern, demagKern, exchKern, MSat, Aex *Quant) Updater {
//	// TODO: verify Aex space-independent
//	CheckSize(dexKern.Size3D(), demagKern.Size3D())
//	CheckSize(dexKern.Size3D(), exchKern.Size3D())
//	return &dexKernUpdater{dexKern, demagKern, exchKern, MSat, Aex}
//}
//
//// Update demag+exchange kernel (cpu)
//func (u *dexKernUpdater) Update() {
//	Debug("Update demagexch")
//
//	dex := u.dexKern.Buffer().List
//	demag := u.demagKern.Buffer().List
//	exch := u.exchKern.Buffer().List
//	MSat := u.MSat.multiplier[0] // Msat may be mask.
//	Aex := u.Aex.Scalar()
//	for i := range dex {
//		// dex = MSat * demag + 2A/µ0MSat * laplacian
//		dex[i] = float32(MSat*float64(demag[i]) + ((2*Aex)/(Mu0*MSat))*float64(exch[i]))
//	}
//}
//
////_____________________________________________________________________ fftkern
//
//// Holds any transformed kernel 
//// as well as the convolution plan that goes with it.
//type fftKernUpdater struct {
//	fftKern *Quant       // that's me!
//	kern    *Quant       //  my dependencies
//	conv    gpu.ConvPlan // TODO: move gpu.ConvPlan into engine?
//}
//
//func newFftKernUpdater(fftKern, kern *Quant) Updater {
//	u := new(fftKernUpdater)
//	u.fftKern = fftKern
//	u.kern = kern
//	return u
//}
//
//func (u *fftKernUpdater) Update() {
//	Debug("Update fftKern")
//	dataSize := GetEngine().GridSize()
//	kernel := make([]*host.Array, 6)
//	for i := range kernel {
//		kernel[i] = (u.kern.Buffer().Component(i))
//	}
//	u.conv.Init(dataSize, kernel, u.fftKern.Array())
//}
//
////____________________________________________________________________ H_dex
//
//// Updates the demag+exchange field in one single convolution
//type hDexUpdater struct {
//	Hdex, m, fftDexKern *Quant
//	conv                *gpu.ConvPlan // points to fftKernDex.updater.conv
//}
//
//func newHDexUpdater(Hdex, m, fftDexKern *Quant) Updater {
//	u := new(hDexUpdater)
//	u.Hdex = Hdex
//	u.m = m
//	u.fftDexKern = fftDexKern
//	u.conv = &(fftDexKern.updater.(*fftKernUpdater).conv)
//	return u
//}
//
//func (u *hDexUpdater) Update() {
//	u.conv.Convolve(&u.m.array, &u.Hdex.array)
//}
