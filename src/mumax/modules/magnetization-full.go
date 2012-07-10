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
func LoadFullMagnetization(e *Engine) {

	LoadMagnetization(e)

	if !e.HasQuant("mf") {
		msat0 := e.AddNewQuant("msat0", SCALAR, MASK, Unit("A/m"), "the initial distribution of the saturation magnetization")
		msat_eq := e.AddNewQuant("msat-eq", SCALAR, MASK, Unit("A/m"), "magnetisation of saturation at zero temperature (equlibrium)")
		mf := e.AddNewQuant("mf", VECTOR, FIELD, Unit(""), "complete magnetization vector reduced by equilibrium value of saturation magnetization")
		mf.SetUpdater(&decomposeMUpdater{mf: mf, msat0: msat0, msat_eq: msat_eq})
		e.Depends("mf", "msat0")
	}
}
