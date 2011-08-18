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
	e.AddDependency("Hd", "m")
	e.AddDependency("Hd", "Msat")

	e.AddVectorField("He")
	e.AddDependency("He", "m")
	e.AddDependency("He", "aexch")

	e.AddVectorField("Hz")
	e.AddDependency("Hz", "t")

	e.AddVectorField("H")
	e.AddDependency("H", "Hd")
	e.AddDependency("H", "He")
	e.AddDependency("H", "Hz")

	e.AddVectorField("torque")
	e.AddDependency("torque", "m")
	e.AddDependency("torque", "H")
	e.AddDependency("torque", "alpha")

	Debug(e.String())
}
