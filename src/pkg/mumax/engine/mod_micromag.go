//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import ()

func (e *Engine) LoadMicromag() {

	e.AddQuant("alpha", SCALAR, MASK)
	e.AddQuant("msat", SCALAR, MASK)
	e.AddQuant("aexch", SCALAR, MASK)

	e.AddQuant("m", VECTOR, FIELD)

	e.AddQuant("H_d", VECTOR, FIELD)
	e.Depends("H_d", "m")
	e.Depends("H_d", "m") // redundant, but should not crash
	e.Depends("H_d", "msat")

	e.AddQuant("H_e", VECTOR, FIELD)
	e.Depends("H_e", "m", "aexch")

	e.AddQuant("H_z", VECTOR, MASK)
	e.Depends("H_z", "t")

	e.AddQuant("H_a", VECTOR, FIELD)
	e.AddQuant("k1", SCALAR, MASK)
	e.AddQuant("k2", SCALAR, MASK)
	e.Depends("H_a", "k1", "k2", "m")

	e.AddQuant("H", VECTOR, FIELD)
	e.Depends("H", "H_d", "H_e", "H_z", "H_a")

	e.AddQuant("torque", VECTOR, FIELD)
	e.Depends("torque", "m", "H", "alpha")

	e.ODE1("m", "torque")
}
