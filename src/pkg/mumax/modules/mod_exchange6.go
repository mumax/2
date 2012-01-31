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
	. "mumax/common"
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
	Aex := e.AddNewQuant("Aex", SCALAR, MASK, Unit("J/m"), "exchange coefficient") // TODO: mask
	Hex := e.AddNewQuant("H_ex", VECTOR, FIELD, Unit("A/m"), "exchange field")
	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_ex")
	e.Depends("H_ex", "Aex", "Msat", "m")
	Hex.SetUpdater(&exch6Updater{m: e.Quant("m"), Aex: Aex, Hex: Hex, Msat: e.Quant("msat")})
}

type exch6Updater struct {
	m, Aex, Hex, Msat *Quant
}

func (u *exch6Updater) Update() {
	e := GetEngine()
	m := u.m
	Aex := u.Aex
	Hex := u.Hex
	Msat := u.Msat

	Aex2_mu0MsatMul := (2 * u.Aex.Multiplier()[0]) / (Mu0 * Msat.Multiplier()[0])
	stream := u.Hex.Array().Stream
	gpu.Exchange6Async(Hex.Array(), m.Array(), Msat.Array(), Aex.Array(), Aex2_mu0MsatMul, e.CellSize(), e.Periodic(), stream)
	stream.Sync()
}
