//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule(&ModLLG{})
}

// Landau-Lifshitz-Gilbert module.
type ModLLG struct{}

func (x ModLLG) Description() string { 
		return "Landau-Lifshitz-Gilbert equation" }

func (x ModLLG) Name() string        { 
		return "llg" }

// The torque array contains reduced Landau-Lifshitz torque τ, in units gamma0*Msat:
//	d m / d t = gamma0 * Msat * τ  
// which is dimensionless and of order 1. Thus the multiplier is gamma0*Msat
// so that the this quantity (array*multiplier) has unit 1/s.
// Note: the unit of gamma0 * Msat is 1/time.
// Thus:
//	τ = (m x h) - α m  x (m x h)
// with:
//	h = H / Msat
func (x ModLLG) Load(e *Engine) {

	e.LoadModule("magnetization")

	e.AddQuant("alpha", SCALAR, MASK, Unit(""), "damping")
	e.AddQuant("gamma", SCALAR, VALUE, Unit("m/As"), "gyromag. ratio")
	e.Quant("gamma").SetScalar(Gamma0)

	e.AddQuant("torque", VECTOR, FIELD, Unit("/s"))
	e.Depends("torque", "m", "H", "alpha", "gamma")
	τ := e.Quant("torque")
	τ.updater = &torqueUpdater{
		τ: e.Quant("torque"),
		m: e.Quant("m"),
		H: e.Quant("H"),
		α: e.Quant("alpha"),
		γ: e.Quant("gamma")}

	e.ODE1("m", "torque")
}

// 
type torqueUpdater struct {
	τ, m, H, α, γ *Quant
}


func (u *torqueUpdater) Update() {
	//Debug("************** H before update torque", u.H.Buffer().Comp[X][0])
	multiplier := u.τ.multiplier
	// must set ALL multiplier components
	γ := u.γ.Scalar()
	if γ == 0 {
		panic(InputErr("gamma should be non-zero"))
	}
	for i := range multiplier {
		multiplier[i] = γ
	}
	gpu.Torque(u.τ.Array(), u.m.Array(), u.H.Array(), u.α.Array(), float32(u.α.Scalar()))
	//Debug("************** H after update torque", u.H.Buffer().Comp[X][0])
}
