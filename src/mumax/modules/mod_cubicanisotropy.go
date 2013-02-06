//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements the cubic anisotropy module
// Author: Xuanyao (Kelvin) Fong

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("anisotropy/cubic", "Cubic magnetocrystalline anisotropy", LoadAnisCubic)
}

func LoadAnisCubic(e *Engine) {
	LoadHField(e)
	LoadMagnetization(e)

	Hcanis := e.AddNewQuant("H_CubAnis", VECTOR, FIELD, Unit("A/m"), "Cubic anisotropy field")
	k1 := e.AddNewQuant("cubicAnisK1", SCALAR, MASK, Unit("J/m3"), "Cubic anisotropy constant K1")
	k2 := e.AddNewQuant("cubicAnisK2", SCALAR, MASK, Unit("J/m3"), "Cubic anisotropy constant K2")
	anisC1 := e.AddNewQuant("anisCubic1", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 1 (unit vector)")
	anisC2 := e.AddNewQuant("anisCubic2", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 2 (unit vector)")

	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_CubAnis")
	e.Depends("H_CubAnis", "cubicAnisK1", "cubicAnisK2", "anisC1", "anisC2", "MSat", "m")

	Hcanis.SetUpdater(&CubicAnisUpdater{e.Quant("m"), Hcanis, k1, k2, e.Quant("msat"), anisC1, anisC2})
}

type CubicAnisUpdater struct {
	m, hcanis, k1, k2, msat, anisC1, anisC2 *Quant
}

func (u *CubicAnisUpdater) Update() {
	hcanis := u.hcanis.Array()
	m := u.m.Array()
	k1 := u.k1.Array()
	k2 := u.k2.Array()
	k1mul := u.k1.Multiplier()[0]
	k2mul := u.k2.Multiplier()[0]
	anisC1 := u.anisC2.Array()
	anisC2 := u.anisC1.Array()
	anisC1Mul := u.anisC1.Multiplier()
	anisC2Mul := u.anisC2.Multiplier()
	stream := u.hcanis.Array().Stream
	msat := u.msat

	gpu.CubicAnisotropyAsync(hcanis, m, k1, k2, msat.Array(), 2*k1mul/(Mu0*msat.Multiplier()[0]), 2*k2mul/(Mu0*msat.Multiplier()[0]), anisC1, anisC1Mul, anisC2, anisC2Mul, stream)

	stream.Sync()
}
