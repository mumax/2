//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// File provides the magnetization quantity
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Load the magnetization and MSat, if not yet present.
func LoadMagnetization(e *Engine) {
	if !e.HasQuant("m") {
		m := e.AddNewQuant("m", VECTOR, FIELD, Unit(""), "magnetization")
		Msat := e.AddNewQuant("Msat", SCALAR, MASK, Unit("A/m"), "saturation magnetization")
		e.Depends("m", "Msat")

		m.SetUpdater(&normUpdater{m: m, Msat: Msat})
	}
}
