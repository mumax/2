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
	RegisterModule("anisotropy/cubic4", "Cubic magnetocrystalline anisotropy (up to fourth order term)", LoadAnisCubic4)
	RegisterModule("anisotropy/cubic6", "Cubic magnetocrystalline anisotropy (up to sixth order term)", LoadAnisCubic6)
	RegisterModule("anisotropy/cubic8", "Cubic magnetocrystalline anisotropy (up to eighth order term)", LoadAnisCubic8)
}

func LoadAnisCubic4(e *Engine) {
	LoadHField(e)
	LoadMagnetization(e)

	Hcanis := e.AddNewQuant("H_Cub4Anis", VECTOR, FIELD, Unit("A/m"), "Fourth order cubic anisotropy field")
	k1 := e.AddNewQuant("cubic4AnisK1", SCALAR, MASK, Unit("J/m3"), "Fourth order cubic anisotropy constant K1")
	anisC1 := e.AddNewQuant("anisCubic4_1", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 1 (unit vector)")
	anisC2 := e.AddNewQuant("anisCubic4_2", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 2 (unit vector)")

	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_Cub4Anis")
	e.Depends("H_Cub4Anis", "cubic4AnisK1", "anisC1", "anisC2", "MSat", "m")

	Hcanis.SetUpdater(&Cubic4AnisUpdater{e.Quant("m"), Hcanis, k1, e.Quant("msat"), anisC1, anisC2})
}

func LoadAnisCubic6(e *Engine) {
	LoadHField(e)
	LoadMagnetization(e)

	Hcanis := e.AddNewQuant("H_Cub6Anis", VECTOR, FIELD, Unit("A/m"), "Sixth order cubic anisotropy field")
	k1 := e.AddNewQuant("cubic6AnisK1", SCALAR, MASK, Unit("J/m3"), "Sixth order cubic anisotropy constant K1")
	k2 := e.AddNewQuant("cubic6AnisK2", SCALAR, MASK, Unit("J/m3"), "Sixth order cubic anisotropy constant K2")
	anisC1 := e.AddNewQuant("anisCubic6_1", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 1 (unit vector)")
	anisC2 := e.AddNewQuant("anisCubic6_2", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 2 (unit vector)")

	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_Cub6Anis")
	e.Depends("H_Cub6Anis", "cubic6AnisK1", "cubic6AnisK2", "anisC1", "anisC2", "MSat", "m")

	Hcanis.SetUpdater(&Cubic6AnisUpdater{e.Quant("m"), Hcanis, k1, k2, e.Quant("msat"), anisC1, anisC2})
}

func LoadAnisCubic8(e *Engine) {
	LoadHField(e)
	LoadMagnetization(e)

	Hcanis := e.AddNewQuant("H_Cub8Anis", VECTOR, FIELD, Unit("A/m"), "Eighth order cubic anisotropy field")
	k1 := e.AddNewQuant("cubic8AnisK1", SCALAR, MASK, Unit("J/m3"), "Eighth order cubic anisotropy constant K1")
	k2 := e.AddNewQuant("cubic8AnisK2", SCALAR, MASK, Unit("J/m3"), "Eighth order cubic anisotropy constant K2")
	k3 := e.AddNewQuant("cubic8AnisK3", SCALAR, MASK, Unit("J/m3"), "Eighth order cubic anisotropy constant K3")
	anisC1 := e.AddNewQuant("anisCubic8_1", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 1 (unit vector)")
	anisC2 := e.AddNewQuant("anisCubic8_2", VECTOR, MASK, Unit(""), "Cubic anisotropy direction 2 (unit vector)")

	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_Cub8Anis")
	e.Depends("H_Cub8Anis", "cubic8AnisK1", "cubic8AnisK2", "cubic8AnisK3", "anisC1", "anisC2", "MSat", "m")

	Hcanis.SetUpdater(&Cubic8AnisUpdater{e.Quant("m"), Hcanis, k1, k2, k3, e.Quant("msat"), anisC1, anisC2})
}

type Cubic4AnisUpdater struct {
	m, hcanis, k1, msat, anisC1, anisC2 *Quant
}

type Cubic6AnisUpdater struct {
	m, hcanis, k1, k2, msat, anisC1, anisC2 *Quant
}

type Cubic8AnisUpdater struct {
	m, hcanis, k1, k2, k3, msat, anisC1, anisC2 *Quant
}

func (u *Cubic4AnisUpdater) Update() {
	hcanis := u.hcanis.Array()
	m := u.m.Array()
	k1 := u.k1.Array()
	k1mul := u.k1.Multiplier()[0]
	anisC1 := u.anisC2.Array()
	anisC2 := u.anisC1.Array()
	anisC1Mul := u.anisC1.Multiplier()
	anisC2Mul := u.anisC2.Multiplier()
	stream := u.hcanis.Array().Stream
	msat := u.msat

	gpu.Cubic6AnisotropyAsync(hcanis, m, k1, msat.Array(), 2*k1mul/(Mu0*msat.Multiplier()[0]), anisC1, anisC1Mul, anisC2, anisC2Mul, stream)

	stream.Sync()
}

func (u *Cubic6AnisUpdater) Update() {
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

	gpu.Cubic6AnisotropyAsync(hcanis, m, k1, k2, msat.Array(), 2*k1mul/(Mu0*msat.Multiplier()[0]), 2*k2mul/(Mu0*msat.Multiplier()[0]), anisC1, anisC1Mul, anisC2, anisC2Mul, stream)

	stream.Sync()
}

func (u *Cubic8AnisUpdater) Update() {
	hcanis := u.hcanis.Array()
	m := u.m.Array()
	k1 := u.k1.Array()
	k2 := u.k2.Array()
	k3 := u.k3.Array()
	k1mul := u.k1.Multiplier()[0]
	k2mul := u.k2.Multiplier()[0]
	k3mul := u.k3.Multiplier()[0]
	anisC1 := u.anisC2.Array()
	anisC2 := u.anisC1.Array()
	anisC1Mul := u.anisC1.Multiplier()
	anisC2Mul := u.anisC2.Multiplier()
	stream := u.hcanis.Array().Stream
	msat := u.msat

	gpu.Cubic8AnisotropyAsync(hcanis, m, k1, k2, k3, msat.Array(), 2*k1mul/(Mu0*msat.Multiplier()[0]), 2*k2mul/(Mu0*msat.Multiplier()[0]), 4*k3mul/(Mu0*msat.Multiplier()[0]), anisC1, anisC1Mul, anisC2, anisC2Mul, stream)

	stream.Sync()
}
