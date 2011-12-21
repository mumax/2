//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("magnetization", "Provides the reduced magnetization and saturation magnetization", LoadMagnetization)
}

func LoadMagnetization(e *Engine) {

	e.AddQuant("m", VECTOR, FIELD, Unit(""), "magnetization")
	e.AddQuant("Msat", SCALAR, MASK, Unit("A/m"), "saturation magnetization")
	e.Depends("m", "Msat")

	m := e.Quant("m")
	Msat := e.Quant("Msat")
	m.SetUpdater(&normUpdater{m: m, Msat: Msat})
}
