//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Provides the Magnetic field
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Loads B if not yet present
func LoadBfield(e *Engine) {
	if e.HasQuant("B") {
		return
	}
	Bfield := e.AddNewQuant("B", VECTOR, FIELD, Unit("T"), "magnetic induction")
	Bfield.SetUpdater(newBfieldUpdater(Bfield))
}

// Updates the E field in a single convolution
// taking into account all possible sources.
type BfieldUpdater struct {
	Bfield    *Quant
	convInput []*gpu.Array // 0, m, μ0J + μ0ε0(∂E/∂t)
	conv      *gpu.Conv73Plan
	// TODO: add B_ext here
}

func newBfieldUpdater(Bfield *Quant) Updater {
	e := GetEngine()
	u := new(BfieldUpdater)
	u.Bfield = Bfield
	// convolution does not have any kernels yet
	// they are added by other modules
	dataSize := e.GridSize()
	logicSize := PadSize(e.GridSize(), e.Periodic())
	u.conv = gpu.NewConv73Plan(dataSize, logicSize)
	u.convInput = make([]*gpu.Array, 7)
	return u
}

func (u *BfieldUpdater) Update() {
	u.conv.Convolve(u.convInput, u.Bfield.Array())
}
