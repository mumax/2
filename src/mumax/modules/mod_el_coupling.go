//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	//~ . "mumax/common"
	. "mumax/engine"
)

const ELfluxName = "Qel"
const ELTiName = "Te"
const ELTjName = "Temp"
const ELcoupName = "Gel"

// Register this module
func init() {
	RegisterModule("temperature/E-L", "Elecron-Lattice coupling", LoadEL)
}

func LoadEL(e *Engine) {
	LoadQinter(e, ELfluxName, ELTiName, ELTjName, ELcoupName)
}
