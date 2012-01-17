//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	//"mumax/host"
)

// Register this module
func init() {
	RegisterModule("current", "Electrical current", LoadCurrent)
}

func LoadCurrent(e *Engine) {

	// here be dragons
	const CPUONLY = true
	const GPU = false

	// Size of all kernels (not FFT'd)
	kernelSize := padSize(e.GridSize(), e.Periodic())

	// charge density
	rho := e.AddNewQuant("rho", SCALAR, FIELD, Unit("C/m3"), "electrical charge density")

	// electrostatic kernel
	elKern := NewQuant("kern_el", VECTOR, kernelSize, FIELD, Unit(""), CPUONLY, "reduced electrostatic kernel")
	e.AddQuant(elKern)
	elKern.SetUpdater(newElKernUpdater(elKern))

	// electric field
	Efield := e.AddNewQuant("E", VECTOR, FIELD, Unit("V/m"), "electrical field")
	e.Depends("E", "rho", "kern_el")
	Efield.SetUpdater(newEfieldUpdater(Efield, rho))

	e.AddNewQuant("j", VECTOR, FIELD, Unit("A/m2"), "electrical current density")
	e.AddNewQuant("sigma", SCALAR, MASK, Unit("1/Ohm*m"), "electrical conductivity")
	e.AddNewQuant("diff_rho", SCALAR, FIELD, Unit("C/m3s"), "time derivative of rho")
	e.Depends("j", "E", "sigma")
	e.Depends("diff_rho", "j")
	e.AddPDE1("rho", "diff_rho")
}

//____________________________________________________________________ E field

// Updates the E field in a single convolution
type EfieldUpdater struct {
	Efield, rho *Quant
	convInput []*gpu.Array
	conv        *gpu.Conv73Plan
}

func newEfieldUpdater(Efield, rho *Quant) Updater {
	u := new(EfieldUpdater)
	u.Efield = Efield
	u.rho = rho
	u.conv = nil
	u.convInput = make([]*gpu.Array, 7)
	u.convInput[0] = rho.Array()
	return u
}

func (u *EfieldUpdater) Update() {
	if u.conv == nil { // todo: kern_el needs to update the conv?
		Debug("Init Electric convolution")
		e := GetEngine()
		dataSize := e.GridSize()
		logicSize := PadSize(e.GridSize(), e.Periodic())
		kernEl := GetEngine().Quant("kern_el").Buffer()
		u.conv = gpu.NewConv73Plan(dataSize, logicSize)
		u.conv.LoadKernel(kernEl, 0, gpu.DIAGONAL, gpu.PUREIMAG)
	}
	u.conv.Convolve(u.convInput, u.Efield.Array())
}

//____________________________________________________________________ electrostatic kernel

// Update demag kernel (cpu)
type elKernUpdater struct {
	kern *Quant // that's me!
}

func newElKernUpdater(elKern *Quant) Updater {
	u := new(elKernUpdater)
	u.kern = elKern
	return u
}

// Update electrostatic kernel (cpu)
func (u *elKernUpdater) Update() {
	e := GetEngine()
	kernsize := padSize(e.GridSize(), e.Periodic())
	// TODO: wisdom
	Log("Calculating electrosatic kernel, may take a moment...")
	PointKernel(kernsize, e.CellSize(), e.Periodic(), u.kern.Buffer())
}
