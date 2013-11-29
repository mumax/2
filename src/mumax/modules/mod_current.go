//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Provides electrical current density
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("current", "Electrical currents", LoadUserDefinedCurrentDensity)
}

// loads the current density
func LoadUserDefinedCurrentDensity(e *Engine) {
	if e.HasQuant("j") {
		Debug("Another electrical current module is already loaded! If it is desired behaviour please ignore this message. Otherwise, please remove all other modules!")
		Debug("Please make sure you add your custom electrical current distibution to the the 'j' quantity")
		return
	}
	e.AddNewQuant("j", VECTOR, MASK, Unit("A/m2"), "electrical current density")
}
