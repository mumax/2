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
	// . "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("exchange6", "6-neighbor ferromagnetic exchange interaction", LoadExch6)
}

func LoadExch6(e *Engine) {
	LoadHField(e)
	LoadFullMagnetization(e)
	lex := e.AddNewQuant("lex", SCALAR, MASK, Unit("J/m"), "Exchange length") // TODO: mask
	Hex := e.AddNewQuant("H_ex", VECTOR, FIELD, Unit("A/m"), "Exchange field")
	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_ex")
	e.Depends("H_ex", "Aex", "Msat0T0", "mf")
	Hex.SetUpdater(&exch6Updater{mf: e.Quant("mf"), lex: lex, Hex: Hex, Msat: e.Quant("Msat0T0")})
}

type exch6Updater struct {
	mf, lex, Hex, Msat *Quant
}

func (u *exch6Updater) Update() {
	e := GetEngine()
	mf := u.mf
	lex := u.lex
	Hex := u.Hex
	Msat := u.Msat

	lexMul2 := lex.Multiplier()[0] * lex.Multiplier()[0]
	stream := u.Hex.Array().Stream
	gpu.Exchange6Async(Hex.Array(), mf.Array(), lex.Array(), lexMul2, Msat.Multiplier()[0], e.CellSize(), e.Periodic(), stream)
	stream.Sync()
}
