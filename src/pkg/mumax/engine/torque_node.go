//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the Landau-Lifshitz torque Quantity.
// Author: Arne Vansteenkiste

import (
	//. "mumax/common"
	"mumax/gpu"
)

// The array contains reduced Landau-Lifshitz torque τ, in units gamma0*Msat:
//	d m / d t = gamma0 * Msat * τ  
// which is dimensionless and of order 1. Thus the multiplier is gamma0*Msat
// so that the this quantity (array*multiplier) has unit 1/s.
// Note: the unit of gamma0 * Msat is 1/time.
// Thus:
//	τ = (m x h) - α m  x (m x h)
// with:
//	h = H / Msat
func (e *Engine) AddTorqueNode() {
	e.AddQuant("torque", VECTOR, FIELD, Unit("/s"))
	e.Depends("torque", "m", "H", "alpha", "Msat", "gamma")
	τ := e.Quant("torque")
	τ.updater = &torqueUpdater{
		τ:    e.Quant("torque"),
		m:    e.Quant("m"),
		H:    e.Quant("H"),
		α:    e.Quant("alpha"),
		Msat: e.Quant("Msat"),
		γ:    e.Quant("gamma")}
}

type torqueUpdater struct {
	τ, m, H, α, Msat, γ *Quant
}

func (u *torqueUpdater) Update() {
	u.τ.multiplier[0] = u.γ.multiplier[0] * u.Msat.multiplier[0]
	gpu.Torque(u.τ.Array(), u.m.Array(), u.H.Array(), u.α.Array(), float32(u.α.Scalar()))
}
