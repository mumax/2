//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements micromagnetic energy terms
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("micromag/energy", "Micromagnetic energy terms **of fields loaded before this module**.", LoadEnergy)
}

func LoadEnergy(e *Engine) {
	LoadHField(e)
	LoadMagnetization(e)

	total := e.AddNewQuant("E", SCALAR, VALUE, Unit("J"), "Sum of all calculated energy terms (this is the total energy only if all relevant energy terms are loaded")
	sumUpd := NewSumUpdater(total).(*SumUpdater)
	total.SetUpdater(sumUpd)

	if e.HasQuant("B_ext") {
		term := LoadEnergyTerm(e, "E_zeeman", "m", "B_ext", -e.CellVolume(), "Zeeman energy")
		Log("Loaded Zeeman energy E_zeeman")
		sumUpd.AddParent(term.Name())
	}

	if e.HasQuant("H_ex") {
		term := LoadEnergyTerm(e, "E_ex", "m", "H_ex", -0.5*e.CellVolume()*Mu0, "Exchange energy")
		Log("Loaded exchange energy E_ex")
		sumUpd.AddParent(term.Name())
	}

	// WARNING: this assumes B is only B_demag.
	if e.HasQuant("B") {
		term := LoadEnergyTerm(e, "E_demag", "m", "B", -0.5*e.CellVolume(), "Demag energy")
		Log("Loaded demag energy E_demag")
		sumUpd.AddParent(term.Name())
	}

	if e.HasQuant("H_anis") {
		term := LoadEnergyTerm(e, "E_anis", "m", "H_anis", -e.CellVolume()*Mu0, "Anisotropy energy")
		Log("Loaded anisotropy energy E_anis")
		sumUpd.AddParent(term.Name())
	}
}

func LoadEnergyTerm(e *Engine, out, in1, in2 string, weight float64, desc string) *Quant {
	Energy := e.AddNewQuant(out, SCALAR, VALUE, Unit("J"), desc)
	e.Depends(out, in1, in2)
	m := e.Quant(in1)
	H := e.Quant(in2)
	Energy.SetUpdater(NewEnergyUpdater(Energy, m, H, e.Quant("msat"), weight))
	return Energy
}

type EnergyUpdater struct {
	*SDotUpdater
	energy *Quant
	msat   *Quant
}

func NewEnergyUpdater(result, m, H, msat *Quant, weight float64) Updater {
	u := new(EnergyUpdater)
	u.SDotUpdater = NewSDotUpdater(result, m, H, weight).(*SDotUpdater)
	u.energy = result
	u.msat = msat
	return u
}

func (u *EnergyUpdater) Update() {
	u.SDotUpdater.Update()
	u.energy.Multiplier()[0] *= u.msat.Multiplier()[0]
}
