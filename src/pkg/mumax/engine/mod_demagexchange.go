//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Combined demag+exchange module
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/gpu"
	"mumax/host"
)

// Register this module
func init() {
	RegisterModule(&ModDemagExch{})
}

// Module for combined calculation of demag + exchange field
// in one single convolution.
type ModDemagExch struct{}

func (x ModDemagExch) Description() string {
	return "combined magnetostatic + exchange field"
}

func (x ModDemagExch) Name() string {
	return "demagexch"
}

func (x ModDemagExch) Load(e *Engine) {
	// dependencies
	e.LoadModule("hfield")
	e.LoadModule("magnetization")
	e.LoadModule("aexchange")

	CPUONLY := true

	// demag kernel 
	kernSize := padSize(e.GridSize(), e.Periodic())
	demagKern := newQuant("kern_d", SYMMTENS, kernSize, FIELD, Unit(""), CPUONLY, "reduced demag kernel (/Msat)")
	e.addQuant(demagKern)
	demagKern.SetUpdater(newDemagKernUpdater(demagKern))

	// exch kernel 
	exchKern := newQuant("kern_ex", SYMMTENS, e.GridSize(), FIELD, Unit("/m2"), CPUONLY, "reduced exchange kernel (Laplacian)")
	e.addQuant(exchKern)
	//exchKern.SetUpdater(&exchKernUpdater{})

	// demag+exchange kernel
	dexKern := newQuant("Kern_dex", SYMMTENS, e.GridSize(), FIELD, Unit("A/m"), CPUONLY, "demag+exchange kernel")
	e.addQuant(dexKern)
	e.Depends("Kern_dex", "kern_d", "kern_ex", "Aex", "MSat")
	dexKern.SetUpdater(&dexKernUpdater{dexKern, demagKern, exchKern, e.Quant("MSat"), e.Quant("Aex")})

	// fft kernel quant
	fftKern := newQuant("~kern_dex", SYMMTENS, e.GridSize(), FIELD, Unit("A/m"), false, "FFT demag+exchange kernel")
	e.addQuant(fftKern)
	e.Depends("~kern_dex", "kern_dex")
	fftKern.SetUpdater(newFftKernUpdater(fftKern, dexKern))

	// demag+exchange field quant
	e.AddQuant("H_dex", VECTOR, FIELD, Unit("A/m"), "demag+exchange field")
	e.Depends("H_dex", "m", "~Kern_dex")
	//Hdex := e.Quant("H_dex")
	//Hdex.SetUpdater(&demagExchHUpdater{Hdex, e.Quant("m"), e.Quant("Msat"), e.Quant("Aex")})

	// add H_dex to total H
	hfield := e.Quant("H")
	sum := hfield.updater.(*SumUpdater)
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
	accuracy := 8
	FaceKernel6(kernsize, e.CellSize(), accuracy, e.Periodic(), u.demagKern.Buffer())
	//Debug("demagkernupdater got", u.demagKern.Buffer())
}

//____________________________________________________________________ exchange kernel

//____________________________________________________________________ demag+exchange kernel

// Update demag+exchange kernel (cpu)
type dexKernUpdater struct {
	dexKern                        *Quant // that's me!
	demagKern, exchKern, MSat, Aex *Quant // my dependencies
}

// Update demag+exchange kernel (cpu)
func (u *dexKernUpdater) Update() {
	Debug("Update demagexch")

	dex := u.dexKern.Buffer().List
	demag := u.demagKern.Buffer().List
	exch := u.exchKern.Buffer().List
	MSat := u.MSat.Scalar()
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
	kern *Quant //  my dependencies
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
	for i:=range kernel{
		kernel[i] = (u.kern.Buffer().Component(i))
	}
	u.conv.Init(dataSize, kernel, u.fftKern.Array())
}

//____________________________________________________________________ H_dex

// Updates the demag+exchange field in one single convolution
type hDexUpdater struct {
	Hdex, m, fftKernDex *Quant
	conv                gpu.ConvPlan // TODO: move gpu.ConvPlan into engine?
	init                bool
}

func newHDexUpdater(Hdex, m, Msat, Aex *Quant) Updater {
	u := new(hDexUpdater)
	//	e := GetEngine()
	//
	//	u.conv.Init(e.GridSize(), Hdex.Comp, Hdex)
	return u
}

func (u *hDexUpdater) Update() {

}
