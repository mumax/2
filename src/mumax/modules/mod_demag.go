//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module for demag field.
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("demag", "Demagnetizing field", LoadDemag)
	RegisterModule("newell_demag", "Demagnetizing field (Newell's formulation)", LoadNewellDemag)
}

// Load demag field
func LoadDemag(e *Engine) {
	LoadMagnetization(e)
	LoadBField(e)
	maxwell.EnableDemag(e.Quant("m"), e.Quant("Msat"))
	e.Depends("B", "m", "Msat")
}

// Load demag field (Newell's formulation)
func LoadNewellDemag(e *Engine) {
	LoadMagnetization(e)
	LoadBField(e)
	maxwell.EnableNewellDemag(e.Quant("m"), e.Quant("Msat"))
	e.Depends("B", "m", "Msat")
}
