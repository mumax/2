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

// Register this module
func init() {
	RegisterModule(&TestModule{})
}

type TestModule struct{}

func (x TestModule) Description() string    { return "Test module" }
func (x TestModule) Name() string           { return "testmodule" }
func (x TestModule) Dependencies() []string { return []string{} }

// Loads a test module.
func (x TestModule) Load(e *Engine) {

	e.AddQuant("m", VECTOR, FIELD, Unit(""), "magnetization")
	e.AddQuant("Msat", SCALAR, MASK, Unit("A/m"), "saturation magn.")
	e.Depends("m", "Msat")

	m := e.Quant("m")
	Msat := e.Quant("Msat")
	m.updater = &normUpdater{m: m, Msat: Msat}

	e.AddQuant("alpha", SCALAR, MASK, Unit(""), "damping")
	e.AddQuant("gamma", SCALAR, VALUE, Unit("m/As"), "gyromag. ratio")
	e.Quant("gamma").SetScalar(Gamma0)

	e.AddQuant("H_z", VECTOR, MASK, Unit("A/m"), "zeeman field")
	e.Depends("H_z", "t")

	e.AddQuant("Aexch", SCALAR, MASK, Unit("J/m"), "exchange coeff.")
	e.AddQuant("H_e", VECTOR, FIELD, Unit("A/m"), "exchange field")
	e.Depends("H_e", "m", "Aexch")

	e.AddQuant("H_d", VECTOR, FIELD, Unit("A/m"), "demag field")
	e.Depends("H_d", "m", "Msat")

	e.AddQuant("H_a", VECTOR, FIELD, Unit("A/m"), "anis. field")
	e.AddQuant("k1", SCALAR, MASK, Unit("J/m3"), "anis. const")
	e.AddQuant("k2", SCALAR, MASK, Unit("J/m3"), "anis. const")
	e.Depends("H_a", "m", "k1", "k2")

	e.AddSumNode("H", "H_z", "H_e", "H_d", "H_a")

	e.ODE1("m", "torque")

	e.AddQuant("rho", VECTOR, FIELD, Unit("C/m3"), "charge density")
	e.AddQuant("j", VECTOR, FIELD, Unit("A/m2"), "current density")
	e.AddQuant("R", SCALAR, FIELD, Unit("mOhm"), "resistivity")
	e.AddQuant("E", VECTOR, FIELD, Unit("V/m"), "electric field")

	e.ODE1("rho", "j")
	e.Depends("j", "E", "R")
	e.Depends("R", "m")
	e.Depends("E", "rho")
	e.Depends("torque", "j")

}
