//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// 6-neighbor exchange interaction
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("exchange6", "6-neighbor ferromagnetic exchange interaction", LoadExch6)
}

func LoadExch6(e *Engine) {
	LoadHField(e)
	LoadMagnetization(e)
	Aex := e.AddNewQuant("Aex", SCALAR, VALUE, Unit("J/m"), "exchange coefficient") // TODO: mask
	Hex := e.AddNewQuant("H_ex", VECTOR, FIELD, Unit("A/m"), "exchange field")
	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_ex")
	e.Depends("H_ex", "Aex", "m")
	Hex.SetUpdater(&exch6Updater{m: e.Quant("m"), Aex: Aex, Hex: Hex})
}

type exch6Updater struct {
	m, Aex, Hex *Quant
}

func (u *exch6Updater) Update() {
	e := GetEngine()
	Aex := float32(u.Aex.Scalar())
	stream := u.Hex.Array().Stream
	gpu.Exchange6Async(u.Hex.Array(), u.m.Array(), Aex, e.CellSize(), e.Periodic(), stream)
	stream.Sync()
}
