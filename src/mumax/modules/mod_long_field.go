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
	RegisterModule("longfield", "The effective field responsible for exchange longitudinal relaxation", LoadLongField)
}

func LoadLongField(e *Engine) {
    LoadHField(e)
    LoadMagnetization(e)
	kappa := e.AddNewQuant("kappa", SCALAR, VALUE, Unit(""), "longitudinal magnetic susceptibility")
	Hlf := e.AddNewQuant("H_lf", VECTOR, FIELD, Unit("A/m"), "longitudinal exchange field")
	msat0 := e.AddNewQuant("msat0", SCALAR, MASK, Unit("A/m"), "the initial distribution of the saturation magnetization")
	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_lf")
	e.Depends("H_lf", "kappa", "Msat", "m", "msat0")
	Hlf.SetUpdater(&LongFieldUpdater{m: e.Quant("m"), kappa: kappa, Hlf: Hlf, Msat: e.Quant("msat"), msat0: msat0})

}

type LongFieldUpdater struct {
	m, kappa, Hlf, Msat, msat0 *Quant
}

func (u *LongFieldUpdater) Update() {
	//e := GetEngine()
	m := u.m
	kappa := u.kappa.Scalar()
	Hlf := u.Hlf
	Msat := u.Msat
    msat0 := u.msat0
	stream := u.Hlf.Array().Stream
	kappa = 0.5 / kappa;
	
	gpu.LongFieldAsync(Hlf.Array(), m.Array(), Msat.Array(), msat0.Array(), kappa, Msat.Multiplier()[0], msat0.Multiplier()[0], stream)
	stream.Sync()
}
