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
	RegisterModule("exchange6",  "6-neighbor ferromagnetic exchange interaction", LoadExch6)
}

func  LoadExch6(e *Engine) {
	e.LoadModule("hfield")
	e.LoadModule("magnetization")
	e.LoadModule("aexchange")
	e.AddQuant("H_ex", VECTOR, FIELD, Unit("A/m"), "exchange field")
	hfield := e.Quant("H")
	sum := hfield.updater.(*SumUpdater)
	sum.AddParent("H_ex")
	e.Depends("H_ex", "Aex", "m")
	Hex := e.Quant("H_ex")
	Hex.updater = &exch6Updater{m: e.Quant("m"), Aex: e.Quant("Aex"), Hex: Hex}
}

type exch6Updater struct {
	m, Aex, Hex *Quant
}

func (u *exch6Updater) Update() {
	panic("To be implemented")
}
