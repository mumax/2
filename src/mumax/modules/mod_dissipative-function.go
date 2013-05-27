//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	"fmt"
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

var inDF = map[string]string{
	"": "",
}

var depsDF = map[string]string{
	"R":     "R",
	"H_eff": "H_eff",
	"m":     "m",
	"msat":  "msat",
}

var outDF = map[string]string{
	"Qmag": "Qmag",
}

// Register this module
func init() {
	args := Arguments{inDF, depsDF, outDF}
	RegisterModuleArgs("dissipative-function", "Signed Dissipative function", args, LoadDFArgs)
}

// There is a problem, since LLB torque is normalized by msat0T0 (zero-temperature value), while LLG torque is normalized by msat
// This has to be explicitly accounted when module is loaded

func LoadDFArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inDF, depsDF, outDF}
	} else {
		arg = args[0]
	}
	//

	// make sure the effective field is in place
	LoadHField(e)

	Qmagn := e.AddNewQuant(arg.Outs("Qmag"), SCALAR, FIELD, Unit("J/(s*m3)"), "The heat flux density from magnetic subsystem to thermal bath")

	if e.HasQuant(arg.Deps("m")) {
		e.Depends(arg.Outs("Qmag"), arg.Deps("m"), arg.Deps("H_eff"), arg.Deps("msat"), arg.Deps("R"))
		Qmagn.SetUpdater(&DFUpdater{
			Qmagn: Qmagn,
			msat:  e.Quant(arg.Deps("msat")),
			m:     e.Quant(arg.Deps("m")),
			Heff:  e.Quant(arg.Deps("H_eff")),
			R:     e.Quant(arg.Deps("R"))})
	} else {
		panic(InputErr(fmt.Sprint("This module is only meaningful for non-conservative dynamics. Please use appropriate model!")))
	}
}

type DFUpdater struct {
	Qmagn *Quant
	msat  *Quant
	m     *Quant
	Heff  *Quant
	R     *Quant
}

func (u *DFUpdater) Update() {

	// Account for msat multiplier, because it is a mask
	u.Qmagn.Multiplier()[0] = u.msat.Multiplier()[0]
	// Account for -0.5 * mu0
	u.Qmagn.Multiplier()[0] *= -0.5 * Mu0
	// Account for multiplier in H_eff
	u.Qmagn.Multiplier()[0] *= u.Heff.Multiplier()[0]
	// From now Qmag = dot(H_eff, torque)
	u.Qmagn.Multiplier()[0] *= u.R.Multiplier()[0]

	gpu.DotSign(u.Qmagn.Array(),
		u.Heff.Array(),
		u.R.Array(),
		u.m.Array())

	// Finally. do Qmag = Qmag * msat(r) to account spatial properties of msat
	if !u.msat.Array().IsNil() {
		gpu.Mul(u.Qmagn.Array(),
			u.Qmagn.Array(),
			u.msat.Array())
	}
}
