//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Implements the time derivative of a quantity
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	"math"
)

// Load time derivative of quant if not yet present
func LoadDerivative(q *Quant) {
	e := GetEngine()
	name := "d" + q.Name() + "_" + "dt"
	if e.HasQuant(name) {
		return
	}
	Assert(q.Kind() == FIELD)
	diff := e.AddNewQuant(name, q.NComp(), FIELD, "("+q.Unit()+")/s", "time derivative of "+q.Name())
	e.Depends(name, q.Name(), "dt", "step")
	updater := newDerivativeUpdater(q, diff)
	diff.SetUpdater(updater)
	diff.SetInvalidator(updater)
}

type derivativeUpdater struct {
	val, diff         *Quant     // original and derived quantities
	lastVal, lastDiff *gpu.Array // previous value for numerical derivative
	lastT             float64    // time of previous value
	lastStep          int        // step of previous value
}

func newDerivativeUpdater(orig, diff *Quant) *derivativeUpdater {
	u := new(derivativeUpdater)
	u.val = orig
	u.diff = diff
	u.lastVal = gpu.NewArray(orig.NComp(), orig.Size3D())  // TODO: alloc only if needed?
	u.lastDiff = gpu.NewArray(orig.NComp(), orig.Size3D()) // TODO: alloc only if needed?
	u.lastT = math.Inf(-1)                                 // so the first time the derivative is taken it will be 0
	u.lastStep = 0                                         //?
	return u
}

func (u *derivativeUpdater) Update() {
	Log("dt update")
}


// called when orig, dt or step changes
func (u *derivativeUpdater) Invalidate() {
	Log("diff invalidate")
	e := GetEngine()
	step := e.step.Scalar()
	if u.lastStep != step {
		u.lastVal.SetTo(u.val.Array())
		u.lastDiff.SetTo(u.diff.Array())
		u.lastT = e.time.Scalar()
		u.lastStep = step
	}
}


//
//	// here be dragons
//	const CPUONLY = true
//	const GPU = false
//
//	// electrostatic kernel
//	kernelSize := padSize(e.GridSize(), e.Periodic())
//	elKern := NewQuant("kern_el", VECTOR, kernelSize, FIELD, Unit(""), CPUONLY, "reduced electrostatic kernel")
//	e.AddQuant(elKern)
//	elKern.SetUpdater(newElKernUpdater(elKern))
//
//	e.Depends("E", "rho", "kern_el")
//}
//
//// Updates electrostatic kernel (cpu)
//type elKernUpdater struct {
//	kern *Quant // that's me!
//}
//
//func newElKernUpdater(elKern *Quant) Updater {
//	u := new(elKernUpdater)
//	u.kern = elKern
//	return u
//}
//
//// Update electrostatic kernel (cpu)
//func (u *elKernUpdater) Update() {
//	e := GetEngine()
//
//	// first update the kernel
//	kernsize := padSize(e.GridSize(), e.Periodic())
//	Log("Calculating electrosatic kernel, may take a moment...")
//	PointKernel(kernsize, e.CellSize(), e.Periodic(), u.kern.Buffer())
//
//	// then also load it into the E field convolution
//	EUpdater := e.Quant("E").GetUpdater().(*EfieldUpdater)
//	kernEl := GetEngine().Quant("kern_el").Buffer()
//	EUpdater.conv.LoadKernel(kernEl, 0, gpu.DIAGONAL, gpu.PUREIMAG)
//	EUpdater.convInput[0] = e.Quant("rho").Array()
//}
