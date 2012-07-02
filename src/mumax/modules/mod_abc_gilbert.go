//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing Slonczewski spin transfer torque.
// Authors: Mykol Dvornik, Graham Rowlands, Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	//"math"
)

// Register this module
func init() {
	RegisterModule("abc_gilbert", "Artificial Gilbert damping term to use with absorbing boudary conditions", LoadABCGilbert)
}

func LoadABCGilbert(e *Engine) {
	LoadMagnetization(e)
	LoadHField(e)
	
	// ============ New Quantities =============
    e.AddNewQuant("abc_alpha", SCALAR, MASK, Unit(""), "Gilbert damping")
    e.AddNewQuant("abc_gamma", SCALAR, VALUE, Unit("m/As"), "gyromag. ratio")
    e.Quant("abc_gamma").SetScalar(Gamma0)
	e.Quant("abc_gamma").SetVerifier(NonZero)
	gilbd := e.AddNewQuant("gilbd", VECTOR, FIELD, Unit("/s"), "Gilbert damping term")

	// ============ Dependencies =============
	e.Depends("gilbd", "m", "H_eff", "abc_alpha", "abc_gamma")

	// ============ Updating the gilbert term =============
	
	gilbd.SetUpdater(&abcGilbertUpdater{
		gilbd: e.Quant("gilbd"),
		    m: e.Quant("m"),
		    H: e.Quant("H_eff"),
		    α: e.Quant("abc_alpha"),
		    γ: e.Quant("abc_gamma")})

	// Add Gilbert Damping to LLG torque
	AddTermToQuant(e.Quant("torque"), gilbd)
}

type abcGilbertUpdater struct {
	gilbd, m, H, α, γ *Quant
}

func (u *abcGilbertUpdater) Update() {
	multiplier := u.gilbd.Multiplier()
	// must set ALL multiplier components
	γ := u.γ.Scalar()
	if γ == 0 {
		panic(InputErr("gamma should be non-zero"))
	}
	for i := range multiplier {
		multiplier[i] = γ
	}
	gpu.Gilbert(u.gilbd.Array(), u.m.Array(), u.H.Array(), u.α.Array(), float32(u.α.Multiplier()[0]))
}
