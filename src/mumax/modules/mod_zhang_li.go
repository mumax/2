//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing Zhang-Li spin transfer torque.
// Authors: Arne Vansteenkiste
//			Rémy Lassalle-Balier

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("zhangli", "INCOMPLETE: Zhang-Li spin transfer torque.", LoadZhangLiTorque)
}

func LoadZhangLiTorque(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...
	e.AddNewQuant("bj", SCALAR, MASK, Unit(""), "Non adiabatic term")
	e.AddNewQuant("cj", SCALAR, MASK, Unit(""), "Adiabatic term")
	e.AddNewQuant("CurrentDensity", VECTOR, FIELD, Unit("A/m2"), "Current density")
}
