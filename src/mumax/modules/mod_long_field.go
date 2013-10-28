//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// The effective field responsible for exchange longitudinal relaxation
// Author: Mykola Dvornik

import (
	. "mumax/engine"
	"mumax/gpu"
)

var inLongField = map[string]string{
	"T": "Teff",
}

var depsLongField = map[string]string{
	"Tc":      "Tc",
	"mf":      "mf",
	"ϰ":       "ϰ",
	"H_eff":   "H_eff",
	"msat0":   "msat0",
	"msat0T0": "msat0T0",
}

var outLongField = map[string]string{
	"H_lf": "H_lf",
}

// Register this module
func init() {
	args := Arguments{inLongField, depsLongField, outLongField}
	RegisterModuleArgs("mfa/longfield", "The effective field responsible for exchange longitudinal relaxation", args, LoadLongFieldArgs)
}

func LoadLongFieldArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inBrillouin, depsBrillouin, outBrillouin}
	} else {
		arg = args[0]
	}
	//

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadTemp(e, arg.Ins("T"))
	LoadKappa(e, arg.Deps("ϰ"))
	LoadMFAParams(e)

	T := e.Quant(arg.Ins("T"))
	Tc := e.Quant(arg.Deps("Tc"))
	mf := e.Quant(arg.Deps("mf"))
	kappa := e.Quant(arg.Deps("ϰ"))
	msat0 := e.Quant(arg.Deps("msat0"))
	msat0T0 := e.Quant(arg.Deps("msat0T0"))
	Hlf := e.AddNewQuant(arg.Outs("H_lf"), VECTOR, FIELD, Unit("A/m"), "longitudinal exchange field")
	e.Depends(arg.Outs("H_lf"), arg.Deps("ϰ"), arg.Deps("msat0"), arg.Deps("mf"), arg.Deps("Tc"), arg.Deps("msat0T0"), arg.Ins("T"))

	hfield := e.Quant(arg.Deps("H_eff"))
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent(arg.Outs("H_lf"))

	Hlf.SetUpdater(&LongFieldUpdater{mf: mf, kappa: kappa, Hlf: Hlf, msat0: msat0, msat0T0: msat0T0, Tc: Tc, T: T})

}

type LongFieldUpdater struct {
	mf, kappa, Hlf, msat0, msat0T0, Tc, T *Quant
}

func (u *LongFieldUpdater) Update() {
	mf := u.mf
	kappa := u.kappa
	Hlf := u.Hlf
	msat0 := u.msat0
	msat0T0 := u.msat0T0
	Tc := u.Tc
	T := u.T
	stream := u.Hlf.Array().Stream
	kappaMul := 2.0 * kappa.Multiplier()[0]

	gpu.LongFieldAsync(Hlf.Array(), mf.Array(), msat0.Array(), msat0T0.Array(), kappa.Array(), Tc.Array(), T.Array(), kappaMul, msat0.Multiplier()[0], msat0T0.Multiplier()[0], Tc.Multiplier()[0], T.Multiplier()[0], stream)
	stream.Sync()
}
