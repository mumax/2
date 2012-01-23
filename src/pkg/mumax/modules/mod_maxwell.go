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
	. "mumax/engine"
)

var maxwell *MaxwellPlan

// Loads E if not yet present
func LoadEField(e *Engine) {
	if e.HasQuant("E") {
		return
	}
	initMaxwell()
	EField := e.AddNewQuant("E", VECTOR, FIELD, Unit("V/m"), "electrical field")
	EExt := e.AddNewQuant("E_ext", VECTOR, MASK, Unit("V/m"), "externally applied electrical field")
	e.Depends("E", "E_ext")
	EField.SetUpdater(newEFieldUpdater())
	maxwell.EExt = EExt
}

// Loads B if not yet present
func LoadBField(e *Engine) {
	if e.HasQuant("B") {
		return
	}
	initMaxwell()
	BField := e.AddNewQuant("B", VECTOR, FIELD, Unit("T"), "magnetic induction")
	BExt := e.AddNewQuant("B_ext", VECTOR, MASK, Unit("T"), "externally applied magnetic induction")
	e.Depends("B", "B_ext")
	BField.SetUpdater(newBFieldUpdater())
	maxwell.BExt = BExt
}

func initMaxwell() {
	e := GetEngine()
	if maxwell == nil {
		maxwell = new(MaxwellPlan)
		maxwell.Init(e.GridSize(), e.PaddedSize())
	}
}
// Updates the E field in a single convolution
// taking into account all possible sources.
type EFieldUpdater struct {

}

func newEFieldUpdater() Updater {
	//	e := GetEngine()
	u := new(EFieldUpdater)
	//	u.EField = EField
	//	// convolution does not have any kernels yet
	//	// they are added by other modules
	//	dataSize := e.GridSize()
	//	logicSize := PadSize(e.GridSize(), e.Periodic())
	//	u.conv = gpu.NewConv73Plan(dataSize, logicSize)
	//	u.convInput = make([]*gpu.Array, 7)
	return u
}

func (u *EFieldUpdater) Update() {

}
// Updates the E field in a single convolution
// taking into account all possible sources.
type BFieldUpdater struct {

}

func newBFieldUpdater() Updater {
	//	e := GetEngine()
	u := new(BFieldUpdater)
	//	u.EField = EField
	//	// convolution does not have any kernels yet
	//	// they are added by other modules
	//	dataSize := e.GridSize()
	//	logicSize := PadSize(e.GridSize(), e.Periodic())
	//	u.conv = gpu.NewConv73Plan(dataSize, logicSize)
	//	u.convInput = make([]*gpu.Array, 7)
	return u
}

func (u *BFieldUpdater) Update() {

}
