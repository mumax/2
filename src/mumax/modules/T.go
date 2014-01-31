//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// File provides various temperature quantities
// Author: Mykola Dvornik

import (
	. "mumax/engine"
)

// Load the temperatures
func LoadT(e *Engine) {
	if !e.HasQuant("Ts") {
		e.AddNewQuant("Ts", SCALAR, FIELD, Unit("K"), "The spin's temperature")
	}
	if !e.HasQuant("Tl") {
		e.AddNewQuant("Tl", SCALAR, FIELD, Unit("K"), "The lattice temperature")
	}
	if !e.HasQuant("Te") {
		e.AddNewQuant("Te", SCALAR, FIELD, Unit("K"), "The electrons temperature")
	}
}
