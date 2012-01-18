//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Provides the Electrical field
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
	"mumax/gpu"
)


// Loads E if not yet present
func LoadEfield(e *Engine) {
	if e.HasQuant("E") {
		return
	}
	Efield := e.AddNewQuant("E", VECTOR, FIELD, Unit("V/m"), "electrical field")
	Efield.SetUpdater(newEfieldUpdater(Efield))
}

// Updates the E field in a single convolution
// taking into account all possible sources.
type EfieldUpdater struct {
	Efield    *Quant
	convInput []*gpu.Array // rho, P, ∂B/∂t
	conv      *gpu.Conv73Plan
}

func newEfieldUpdater(Efield *Quant) Updater {
	e := GetEngine()
	u := new(EfieldUpdater)
	u.Efield = Efield
	// convolution does not have any kernels yet
	// they are added by other modules
	dataSize := e.GridSize()
	logicSize := PadSize(e.GridSize(), e.Periodic())
	u.conv = gpu.NewConv73Plan(dataSize, logicSize)
	u.convInput = make([]*gpu.Array, 7)
	return u
}

func (u *EfieldUpdater) Update() {
	u.conv.Convolve(u.convInput, u.Efield.Array())
}
