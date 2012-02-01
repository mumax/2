//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Provides electrical current density
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("current", "Electrical currents", LoadCurrent)
}

func LoadCurrent(e *Engine) {
	if e.HasQuant("j") {
		return
	}
	LoadCoulomb(e)
	e.AddNewQuant("j", VECTOR, FIELD, Unit("A/m2"), "electrical current density")
	e.AddNewQuant("r", SCALAR, MASK, Unit("Ωm"), "electrical resistivity")
}
