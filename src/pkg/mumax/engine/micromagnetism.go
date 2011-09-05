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
	e.AddScalar("Msat")
	e.AddScalar("aexch")

	e.AddVectorField("m")

	e.AddVectorField("Hd")
	e.Depends("Hd", "m")
	e.Depends("Hd", "Msat")

	e.AddVectorField("He")
	e.Depends("He", "m")
	e.Depends("He", "aexch")

	e.AddVectorField("Hz")
	e.Depends("Hz", "t")

	e.AddVectorField("H")
	e.Depends("H", "Hd")
	e.Depends("H", "He")
	e.Depends("H", "Hz")

	e.AddVectorField("torque")
	e.Depends("torque", "m")
	e.Depends("torque", "H")
	e.Depends("torque", "alpha")

	Debug(e.String())
}
