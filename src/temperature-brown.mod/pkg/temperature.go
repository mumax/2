//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package temperature_brown

// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)



// Register this module
func init() {
	RegisterModule(ModTempBrown(0))
}

type ModTempBrown int

func (x ModTempBrown) Description() string {
	return "Thermal fluctuating field according to Brown."
}

func (x ModTempBrown) Name() string {
	return "temperature/brown"
}

func (x ModTempBrown) Load(e *Engine) {

	// TODO: make it a mask so we can have temperature gradients
	e.AddQuant("Temp", SCALAR, VALUE, Unit("K"), "Temperature")
	e.AddQuant("H_therm", VECTOR, FIELD, Unit("A/m"), "Thermal fluctuating field")
	e.Depends("H_therm", "Temp", "Step")
	
	e.LoadModule("hfield")
	hfield := e.Quant("H")
	sum := hfield.GetUpdater().(*SumUpdater)
	sum.AddParent("H_therm")

}



