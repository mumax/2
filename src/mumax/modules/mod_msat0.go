//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// File provides the (temperature-dependant) equilibrium staturation magnetization quantity
// Author: Mykola Dvornik

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("msat0", "Temperature dependant equilibrium saturation magnetization", LoadLongField)
}

// Load the magnetization and MSat, if not yet present.
func LoadMsatEq(e *Engine) {

	if !e.HasQuant("msat0") {
		e.AddNewQuant("msat0", SCALAR, MASK, Unit("A/m"), "the initial distribution of the saturation magnetization")
		e.AddNewQuant("msat0T", SCALAR, MASK, Unit("A/m"), "the equlibrium saturation magnetization at a given temperature")
		e.Depends("msat0T", "Ts", "msat0")
		
	}
}

