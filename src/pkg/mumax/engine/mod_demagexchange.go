//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Combined demag+exchange module
// Author: Arne Vansteenkiste

import ()

// Register this module
func init() {
	RegisterModule(&ModDemagExch{})
}

type ModDemagExch struct{}

func (x ModDemagExch) Description() string {
	return "combined magnetostatic + exchange field"
}

func (x ModDemagExch) Name() string {
	return "demagexch"
}

func (x ModDemagExch) Load(e *Engine) {
	e.LoadModule("hfield")
	e.LoadModule("magnetization")
	e.LoadModule("aexchange")

	e.AddQuant("H_dex", VECTOR, FIELD, Unit("A/m"), "demag+exchange field")
	hfield := e.Quant("H")
	sum := hfield.updater.(*SumUpdater)
	sum.AddParent("H_dex")
	e.Depends("H_dex", "Aex", "m")
	Hdex := e.Quant("H_dex")
	Hdex.updater = &demagexchUpdater{m: e.Quant("m"), Aex: e.Quant("Aex"), Hdex: Hdex}
}

type demagexchUpdater struct {
	m, Aex, Hdex *Quant
}

func (u *demagexchUpdater) Update() {
	println("To be implemented")
}
