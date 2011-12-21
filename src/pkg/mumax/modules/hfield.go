//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// File provides the "H" quantity.
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Loads the "H" quantity if it is not already present.
func LoadHField(e *Engine) {
	if !e.HasQuant("H") {
		e.AddNewQuant("H", VECTOR, FIELD, Unit("A/m"), "magnetic field")
		q := e.Quant("H")
		q.SetUpdater(NewSumUpdater(q))
	}
}
