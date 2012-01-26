//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Implements the time derivative of a quantity
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/gpu"
	"math"
)

// TODO: multipliers??

// Load time derivative of quant if not yet present
func (e *Engine) AddTimeDerivative(q *Quant) {
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
	Log("diff update")

	// not stable:
	//²	// f'(t) = 2(f(t)-f(0))/(t1-t0) - f'(t0)
	//²	t := engine.time.Scalar()
	//²	dt := t - u.lastT
	//²	Assert(dt >= 0)
	//²	diff := u.diff.Array()
	//²	val := u.val.Array()
	//²	if dt == 0 {
	//²		Log("dt==0")
	//²		diff.CopyFromDevice(u.lastDiff)
	//²	} else {
	//²		Log("dt!=0")
	//²		gpu.LinearCombination3Async(diff, val, float32(2/dt), u.lastVal, -float32(2/dt), u.lastDiff, -1, diff.Stream)
	//²		diff.Sync()
	//²	}


	t := engine.time.Scalar()
	dt := t - u.lastT
	Assert(dt >= 0)
	diff := u.diff.Array()
	val := u.val.Array()
	if dt == 0 {
		Log("dt==0")
		diff.CopyFromDevice(u.lastDiff)
	} else {
		Log("dt!=0")
		gpu.LinearCombination2Async(diff, val, float32(1/dt), u.lastVal, -float32(1/dt), diff.Stream)
		diff.Sync()
	}

}

// called when orig, dt or step changes
// TODO: pre-invalidator
func (u *derivativeUpdater) Invalidate() {
	e := GetEngine()
	step := int(e.step.Scalar())
	if u.lastStep != step {
		Log("diff invalidate")
		u.Update() // TODO: only if needed !!
		u.lastVal.CopyFromDevice(u.val.Array())
		u.lastDiff.CopyFromDevice(u.diff.Array())
		u.lastT = e.time.Scalar()
		u.lastStep = step
	}
}
