//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module for Coulomb's law.
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("coulomb", "Coulomb's law", LoadCoulomb)
}

// Load Coulomb's law
func LoadCoulomb(e *Engine) {
	LoadEField(e)
	if e.HasQuant("rho") {
		return
	}
	rho := e.AddNewQuant("rho", SCALAR, FIELD, Unit("C/m3"), "electrical charge density")
	e.Depends("E", "rho")
	initMaxwell()
	maxwell.EnableCoulomb(rho, e.Quant("E"))
}
