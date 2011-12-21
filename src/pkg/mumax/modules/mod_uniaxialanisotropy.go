//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements the uniaxial anisotropy module
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("anisotropy/uniaxial", "INCOMPLETE: Uniaxial magnetocrystalline anisotropy", LoadAnisUniaxial)
}

func LoadAnisUniaxial(e *Engine) {
	LoadHField(e)
	e.AddNewQuant("H_anis", VECTOR, FIELD, Unit("A/m"), "uniaxial anisotropy field")
	e.AddNewQuant("k1", SCALAR, MASK, Unit("J/m3"), "uniaxial anisotropy constant K1") //TODO: Ku1
	e.AddNewQuant("k2", SCALAR, MASK, Unit("J/m3"), "uniaxial anisotropy constant K2")
	hfield := e.Quant("H")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_anis")
	e.Depends("H_anis", "k1", "k2")
	//e.Depends("k1", "t")
	//e.Depends("k2", "t")
}
