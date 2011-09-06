//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import ()

func (e *Engine) LoadMicromag() {

	e.AddScalar("alpha")
	e.AddScalar("msat")
	e.AddScalar("aexch")

	e.AddVectorField("m")

	e.AddVectorField("H_d")
	e.Depends("H_d", "m")
	e.Depends("H_d", "m") // redundant, but should not crash
	e.Depends("H_d", "msat")

	e.AddVectorField("H_e")
	e.Depends("H_e", "m")
	e.Depends("H_e", "aexch")

	e.AddVectorField("H_z")
	e.Depends("H_z", "t")

	e.AddVectorField("H_a")
	e.AddScalar("k1")
	e.AddScalar("k2")
	e.Depends("H_a", "k1")
	e.Depends("H_a", "k2")
	e.Depends("H_a", "m")

	e.AddVectorField("H")
	e.Depends("H", "H_d")
	e.Depends("H", "H_e")
	e.Depends("H", "H_z")
	e.Depends("H", "H_a")

	e.AddVectorField("torque")
	e.Depends("torque", "m")
	e.Depends("torque", "H")
	e.Depends("torque", "alpha")

	e.ODE1("m", "torque")
}
