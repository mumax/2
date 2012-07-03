//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Provides electrical current density
// Author: Mykola Dvornik

import (
	. "mumax/engine"
	"fmt"
	. "mumax/common"
)

// Register this module
func init() {
	RegisterModule("current-sum", "Sum of electrical currents", LoadUserDefinedCurrentDensitySum)
}

// loads the current density
func LoadUserDefinedCurrentDensitySum(e *Engine) {
	if e.HasQuant("j") {
		panic(InputErr(fmt.Sprint("You have already loaded another electrical current module, please make sure that this module is loaded first!")))
	}
	
	j := e.AddNewQuant("j", VECTOR, FIELD, Unit("A/m2"), "sum of custom electrical current densities")
	j.SetUpdater(NewSumUpdater(j))	
}
