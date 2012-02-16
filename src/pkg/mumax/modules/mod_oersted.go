//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module for Oersted fields
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("oersted", "Oersted field of electrical current", LoadOersted)
}

//TODO: j with multiplier Mu0 to get order of magnitude right?
func LoadOersted(e *Engine) {
	LoadBField(e)
	LoadUserDefinedCurrentDensity(e)
	e.Depends("B", "j")
	maxwell.EnableOersted(e.Quant("j"))
}
