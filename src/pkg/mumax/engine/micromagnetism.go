//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
)

func (e *Engine) InitMicromagnetism() {
	Debug("engine.InitMicromagnetism")

	e.AddScalar("t")

	e.AddScalar("alpha")
	e.AddScalar("msat")
	e.AddScalar("aexch")

	e.AddVectorField("m")

	e.AddVectorField("h_d")
	e.Depends("h_d", "m")
	e.Depends("h_d", "msat")

	e.AddVectorField("h_e")
	e.Depends("h_e", "m")
	e.Depends("h_e", "aexch")

	e.AddVectorField("h_z")
	e.Depends("h_z", "t")

	e.AddVectorField("h")
	e.Depends("h", "h_d")
	e.Depends("h", "h_e")
	e.Depends("h", "h_z")

	e.AddVectorField("torque")
	e.Depends("torque", "m")
	e.Depends("torque", "h")
	e.Depends("torque", "alpha")

	Debug(e.String())
}
