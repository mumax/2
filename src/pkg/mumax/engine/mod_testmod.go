//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import ()

// Loads a test module.
func (e *Engine) LoadTest() {

	e.AddQuant("m", VECTOR, FIELD, Unit(""), "magnetization")
	e.AddQuant("Msat", VECTOR, FIELD, Unit("A/m"), "saturation magn.")

	e.AddQuant("alpha", SCALAR, MASK, Unit(""), "damping")

	e.AddQuant("h_z", VECTOR, MASK, Unit("A/m"), "zeeman field")
	e.Depends("h_z", "t")

	//e.AddQuant("h", VECTOR, FIELD, "red. field")
	//e.Depends("h", "h_z")
	e.AddSumNode("h", "h_z")

	e.AddTorqueNode()

	e.ODE1("m", "torque")
}
