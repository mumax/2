//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import ()

// Register this module
func init() {
	RegisterModule(&Magnetization{})
}

type Magnetization struct{}

func (x Magnetization) Description() string {
	return "m: normalized magnetization, mSat: saturation magnetization [A/m]"
}

func (x Magnetization) Name() string {
	return "magnetization"
}

func (x Magnetization) Load(e *Engine) {

	e.AddQuant("m", VECTOR, FIELD, Unit(""), "magnetization")
	e.AddQuant("Msat", SCALAR, MASK, Unit("A/m"), "saturation magn.")
	e.Depends("m", "Msat")

	m := e.Quant("m")
	Msat := e.Quant("Msat")
	m.updater = &normUpdater{m: m, Msat: Msat}
}
