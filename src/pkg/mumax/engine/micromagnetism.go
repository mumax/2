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

	e.AddVectorField("H_d")
	e.Depends("H_d", "m")
	e.Depends("H_d", "m") // redundant
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

	e.AddScalarField("e_e")
	e.Depends("e_e", "m")
	e.Depends("e_e", "H_e")
	e.AddScalarField("E_e")
	e.Depends("E_e", "e_e")

	e.AddScalarField("e_d")
	e.Depends("e_d", "m")
	e.Depends("e_d", "H_d")
	e.AddScalarField("E_d")
	e.Depends("E_d", "e_d")

	e.AddScalarField("e_z")
	e.Depends("e_z", "m")
	e.Depends("e_z", "H_z")
	e.AddScalarField("E_z")
	e.Depends("E_z", "e_z")

	e.AddScalarField("e_a")
	e.Depends("e_a", "m")
	e.Depends("e_a", "H_a")
	e.AddScalarField("E_a")
	e.Depends("E_a", "e_a")

	e.AddScalarField("e")
	e.Depends("e", "e_a")
	e.Depends("e", "e_z")
	e.Depends("e", "e_e")
	e.Depends("e", "e_d")

	e.AddScalarField("E")
	e.Depends("E", "E_a")
	e.Depends("E", "E_z")
	e.Depends("E", "E_e")
	e.Depends("E", "E_d")

	e.AddVectorField("m_norm")
	e.Depends("m", "m_norm")

	e.AddVectorField("j")
	e.Depends("j", "t")
	e.Depends("torque", "j")

	Debug(e.String())
}
