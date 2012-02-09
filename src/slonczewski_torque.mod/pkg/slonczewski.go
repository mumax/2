//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package slonczewski_torque

// Module implementing Slonczewski spin transfer torque.
// Authors: Graham Rowlands

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("slonczewski", "Slonczewski spin transfer torque.", LoadSlonczewskiTorque)
}

func LoadSlonczewskiTorque(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...
	//e.LoadModule("current") // Should eventually intgrate with maxwell stuff
	e.AddNewQuant("aj", SCALAR, VALUE, Unit("unitless"), "In-Plane term")
	e.AddNewQuant("bj", SCALAR, VALUE, Unit("unitless"), "Field-Like term")
	e.AddNewQuant("p", VECTOR, FIELD, Unit("unitless"), "Polarization Vector")
	e.AddNewQuant("Pol", SCALAR, VALUE, Unit("unitless"), "Polarization Efficiency")
	e.AddNewQuant("CurrentDensity", SCALAR, FIELD, Unit("A/m2"), "Current density")
}
