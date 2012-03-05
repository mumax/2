//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Provides the Electrical and Magnetic field
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
)

var (
	maxwellv MaxwellPlan
	maxwell  *MaxwellPlan = &maxwellv
)

// Loads E if not yet present
func LoadEField(e *Engine) {
	if e.HasQuant("E") {
		return
	}
	EField := e.AddNewQuant("E", VECTOR, FIELD, Unit("V/m"), "electrical field")
	EExt := e.AddNewQuant("E_ext", VECTOR, MASK, Unit("V/m"), "externally applied electrical field")
	e.Depends("E", "E_ext")
	EField.SetUpdater(newEFieldUpdater())
	maxwell.EExt = EExt
	maxwell.E = EField
}

// Loads B if not yet present
func LoadBField(e *Engine) {
	if e.HasQuant("B") {
		return
	}
	BField := e.AddNewQuant("B", VECTOR, FIELD, Unit("T"), "magnetic induction")
	BExt := e.AddNewQuant("B_ext", VECTOR, MASK, Unit("T"), "externally applied magnetic induction")
	e.Depends("B", "B_ext")
	BField.SetUpdater(newBFieldUpdater())
	maxwell.BExt = BExt
	maxwell.B = BField
	// Add B/mu0 to H_eff
	if e.HasQuant("H_eff") {
		sum := e.Quant("H_eff").Updater().(*SumUpdater)
		sum.MAddParent("B", 1/Mu0)
	}
}

// Updates the E field in a single convolution
// taking into account all possible sources.
type EFieldUpdater struct {

}

func newEFieldUpdater() Updater {
	u := new(EFieldUpdater)
	return u
}

func (u *EFieldUpdater) Update() {
	maxwell.UpdateE()
}
// Updates the E field in a single convolution
// taking into account all possible sources.
type BFieldUpdater struct {

}

func newBFieldUpdater() Updater {
	u := new(BFieldUpdater)
	return u
}

func (u *BFieldUpdater) Update() {
	maxwell.UpdateB()
}
