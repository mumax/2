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

	if e.HasQuant("B_ext"){
		LoadEnergyTerm(e, "E_zeeman", "m", "B_ext", -e.CellVolume(), "Zeeman energy")
	}

	if e.HasQuant("H_ex"){
		LoadEnergyTerm(e, "E_ex", "m", "H_ex", -0.5*e.CellVolume()*Mu0, "Exchange energy")
	}
}

func LoadEnergyTerm(e *Engine, out, in1, in2 string, weight float64, desc string){
	Energy := e.AddNewQuant(out, SCALAR, VALUE, Unit("J"), desc)
	e.Depends(out, in1, in2)
	m := e.Quant(in1)
	H := e.Quant(in2)
	Energy.SetUpdater(NewSDotUpdater(Energy, m, H, weight))
}

