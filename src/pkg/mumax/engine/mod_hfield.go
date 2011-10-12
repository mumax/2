//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import ()

// Register this module
func init() {
	RegisterModule(&ModHField{})
}

// Module for the total field H. Other modules like H_demag, H_anis, H_ext have
// to add their field to the sum make by H.
type ModHField struct{}

func (x ModHField) Description() string {
	return "H: total magnetic field [A/m]"
}

func (x ModHField) Name() string {
	return "hfield"
}

func (x ModHField) Load(e *Engine) {
	e.AddQuant("H", VECTOR, FIELD, Unit("A/m"), "magnetic field")
	q := e.Quant("H")
	// do we store H, or h + multiplier?
	panic("need to set multiplier here? somewhere else?")
	q.updater = &SumUpdater{q}
}
