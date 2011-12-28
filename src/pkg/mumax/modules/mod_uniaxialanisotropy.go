//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements the uniaxial anisotropy module
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("anisotropy/uniaxial", "Uniaxial magnetocrystalline anisotropy", LoadAnisUniaxial)
}

func LoadAnisUniaxial(e *Engine) {
	LoadHField(e)

	Hanis := e.AddNewQuant("H_anis", VECTOR, FIELD, Unit("A/m"), "uniaxial anisotropy field")
	ku1 := e.AddNewQuant("Ku1", SCALAR, MASK, Unit("J/m3"), "uniaxial anisotropy constant K1")
	ku2 := e.AddNewQuant("Ku2", SCALAR, MASK, Unit("J/m3"), "uniaxial anisotropy constant K2")
	anisU := e.AddNewQuant("anisU", VECTOR, MASK, Unit(""), "uniaxial anisotropy direction (unit vector)")

	hfield := e.Quant("H")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_anis")
	e.Depends("H_anis", "Ku1", "Ku2", "anisU")

	Hanis.SetUpdater(&UniaxialAnisUpdater{e.Quant("m"), Hanis, ku1, ku2, anisU})
}

type UniaxialAnisUpdater struct {
	m, hanis, ku1, ku2, anisU *Quant
}

func (u *UniaxialAnisUpdater) Update() {
	gpu.UniaxialAnisotropyAsync(
		u.hanis.Array(),
		u.m.Array(),
		u.ku1.Array(), u.ku1.Multiplier()[0],
		u.ku2.Array(), u.ku2.Multiplier()[0],
		u.anisU.Array(), u.anisU.Multiplier(),
		gpu.STREAM0)
	//u.hanis.Array().Stream)
	u.hanis.Array().Stream.Sync()
}
